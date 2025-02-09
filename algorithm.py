import cv2
from typing import List, Optional
from main import get_available_spots
from util import get_parking_spots_bboxes

def normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value) if min_value != max_value else 0.0

def find_optimal_spot(available_spots: List[dict], weight_entrance=0.4, weight_entry=0.3, weight_exit=0.3) -> Optional[dict]:
    if not available_spots:
        return None

    min_d1, max_d1 = min(s['distance_from_entrance'] for s in available_spots), max(s['distance_from_entrance'] for s in available_spots)
    min_d2, max_d2 = min(s['distance_from_entry'] for s in available_spots), max(s['distance_from_entry'] for s in available_spots)
    min_d3, max_d3 = min(s['distance_to_exit'] for s in available_spots), max(s['distance_to_exit'] for s in available_spots)

    best_spot, best_score = None, float('inf')
    for spot in available_spots:
        score = (weight_entrance * normalize(spot['distance_from_entrance'], min_d1, max_d1) +
                 weight_entry * normalize(spot['distance_from_entry'], min_d2, max_d2) +
                 weight_exit * normalize(spot['distance_to_exit'], min_d3, max_d3))
        if score < best_score:
            best_score, best_spot = score, spot
    return best_spot

def process_video_in_real_time(mask_path='./mask_1920_1080.png', video_path='./samples/parking_1920_1080_loop.mp4'):
    # Load mask and video
    mask = cv2.imread(mask_path, 0)
    cap = cv2.VideoCapture(video_path)
    
    # Get parking spots from connected components
    connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    spots = get_parking_spots_bboxes(connected_components)
    entry_points = [(95, 58), (1464, 64), (109, 1074)]
    
    frame_count = 0
    previous_frame = None
    optimal_spot= None

    # Process video frame-by-frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check the optimal spot only once every 30 frames
        if frame_count % 150 == 0:
            available_spots = get_available_spots(frame, spots, previous_frame, diff_threshold=5)
            optimal_spot = find_optimal_spot(available_spots)
            print(optimal_spot)
            previous_frame = frame.copy()
        
        # Highlight entry points in red
        for point in entry_points:
            cv2.circle(frame, point, 10, (0, 0, 255), -1)
        
        # Draw all parking spots and highlight the optimal one
        for spot_indx, spot in enumerate(spots):
            x1, y1, w, h = spot[:4]
            spot_id = f'Spot_{spot_indx + 1}'
            color = (0, 255, 0) if optimal_spot and spot_id == optimal_spot['spot_id'] else (150, 150, 150)
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)
        
        # Display the current frame
        cv2.imshow('Parking Lot', frame)
        
        # Increment the frame count
        frame_count += 1
        
        # Break loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Call the function to start processing the video in real time
    process_video_in_real_time()

# Ensure the script runs only when executed directly
if __name__ == '__main__':
    main()
