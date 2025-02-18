class ObjectTracker:
    def __init__(self, frame_check_threshold, fire_smoke_frame_check_threshold, alert_threshold, warning_ratio, danger_ratio):
        self.tracked_objects = {}
        self.frame_check_threshold = frame_check_threshold
        self.fire_smoke_frame_check_threshold = fire_smoke_frame_check_threshold
        self.alert_threshold = alert_threshold
        self.warning_ratio = warning_ratio
        self.danger_ratio = danger_ratio

    def track_object(self, class_name, area_ratio):
        if class_name not in self.tracked_objects:
            self.tracked_objects[class_name] = {
                'area_ratios': [],
                'alert_level': '관심',
                'frames_since_first_detection': 0
            }
        else:
            self.tracked_objects[class_name]['alert_level'] = '관심'

        self.tracked_objects[class_name]['area_ratios'].append(area_ratio)
        self.tracked_objects[class_name]['frames_since_first_detection'] += 1

        if self.tracked_objects[class_name]['alert_level'] == '관심':
            if len(self.tracked_objects[class_name]['area_ratios']) >= self.frame_check_threshold:
                recent_ratios = self.tracked_objects[class_name]['area_ratios'][-self.frame_check_threshold:]
                ratio_change = recent_ratios[-1] - recent_ratios[0]

                if ratio_change >= self.danger_ratio:
                    self.tracked_objects[class_name]['alert_level'] = '위험'
                    print(f"[위험] {class_name} detected with area ratio increase to {recent_ratios[-1]:.2f}")
                elif ratio_change >= self.warning_ratio:
                    self.tracked_objects[class_name]['alert_level'] = '경고'
                    print(f"[경고] {class_name} detected with area ratio increase to {recent_ratios[-1]:.2f}")

        if class_name in ['fire', 'smoke']:
            if self.tracked_objects[class_name]['frames_since_first_detection'] > self.fire_smoke_frame_check_threshold:
                print(f"[경고] {class_name} detected for more than {self.fire_smoke_frame_check_threshold} frames")
