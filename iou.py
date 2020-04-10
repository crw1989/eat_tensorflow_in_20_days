wh = np.expand_dims(wh, -2)  # shape:(N,1,2)
wh_max = wh // 2
wh_min = -wh_max
max_xy = np.maximum(anchors_min, wh_min)
min_xy = np.minimum(anchor_max, wh_max)
intersect_area = (min_xy - max_xy)[..., 0] * (min_xy - max_xy)[..., 1]
area_anchor = anchors[..., 0] * anchors[..., 1]
box_area = wh[..., 0] * wh[..., 1]
iou = intersect_area / (area_anchor + box_area - intersect_area)
idx = np.argmax(iou, axis=-1)
