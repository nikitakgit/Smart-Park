from util import get_parking_spots_bboxes, empty_or_not,calc_diff

def get_available_spots(frame, spots, previous_frame, diff_threshold):
    available_spots = []

    # Calculate differences only if a previous frame is provided
    diffs = [0] * len(spots) if previous_frame is not None else None

    for spot_indx, spot in enumerate(spots):
        x1, y1, w, h = spot[:4]
        spot_crop = frame[y1:y1 + h, x1:x1 + w, :]

        # Calculate difference only if previous frame is provided
        if previous_frame is not None:
            prev_spot_crop = previous_frame[y1:y1 + h, x1:x1 + w, :]
            diffs[spot_indx] = calc_diff(spot_crop, prev_spot_crop)
        
        # Only check occupancy if difference exceeds the threshold
        if previous_frame is None or diffs[spot_indx] > diff_threshold:
            spot_status = empty_or_not(spot_crop)
            if spot_status:  # If the spot is empty
                d1, d2, d3 = spot[4:7]  # Distances from entry and exit points
                available_spots.append({
                    'spot_id': f'Spot_{spot_indx + 1}',
                    'distance_from_entrance': d1,
                    'distance_from_entry': d2,
                    'distance_to_exit': d3,
                })

    return available_spots