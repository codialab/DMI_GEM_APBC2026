import numpy as np

# # example to use the roi_masks:
# roi = roi_masks["Gastrocnemius"]["w3"][47]
# if roi is not None:
#     value = np.nanmean(meta_map[m_idx][roi])

NX, NY = 16, 16

def single_voxel_mask(xy):
    """
    xy: [x, y] or None
    returns: (16,16) boolean mask or None
    """
    if xy is None:
        return None
    mask = np.zeros((NX, NY), dtype=bool)
    x0, y0 = xy
    x, y = x0 - 1, y0 - 1  # convert to 0-based indexing
    mask[x, y] = True
    return mask

# note that the coordinates are 1-based (as in Matlab)
def f_get_roi_masks():
    roi_masks = {

        "Knochenmark": {
            "basal": {
                45: single_voxel_mask([5,7]),
                47: single_voxel_mask([8,7]),
                48: single_voxel_mask([8,7]),
                49: single_voxel_mask([8,7]),
                50: single_voxel_mask([8,7]),
                51: single_voxel_mask([7,7]),
                52: single_voxel_mask([8,7]),
                53: single_voxel_mask([8,7]),
                54: single_voxel_mask([6,7]),
                55: single_voxel_mask([8,7]),
                56: single_voxel_mask([7,7]),
            },
            "w3": {
                45: single_voxel_mask([7,7]),
                47: single_voxel_mask([7,7]),
                48: single_voxel_mask([9,7]),
                49: single_voxel_mask([8,7]),
                50: single_voxel_mask([8,7]),
                51: single_voxel_mask([7,7]),
                52: single_voxel_mask([9,7]),
                53: single_voxel_mask([7,7]),
                54: single_voxel_mask([8,7]),
                55: single_voxel_mask([6,7]),
                56: single_voxel_mask([8,7]),
            },
            "w6": {
                45: single_voxel_mask([6,7]),
                47: single_voxel_mask([8,7]),
                48: single_voxel_mask([8,7]),
                49: single_voxel_mask([6,7]),
                50: single_voxel_mask([7,7]),
                51: single_voxel_mask([8,7]),
                52: single_voxel_mask([8,8]),
                53: None,
                54: single_voxel_mask([9,7]),
                55: single_voxel_mask([7,7]),
                56: single_voxel_mask([6,7]),
            },
            "w9": {
                45: single_voxel_mask([8,8]),
                47: single_voxel_mask([6,7]),
                48: single_voxel_mask([5,7]),
                49: single_voxel_mask([9,7]),
                50: single_voxel_mask([6,8]),
                51: single_voxel_mask([7,7]),
                52: single_voxel_mask([8,7]),
                53: None,
                54: single_voxel_mask([7,7]),
                55: single_voxel_mask([6,7]),
                56: single_voxel_mask([6,7]),
            },
        },

        "TibialisAnt": {
            "basal": {
                45: single_voxel_mask([4,9]),
                47: single_voxel_mask([6,8]),
                48: single_voxel_mask([6,8]),
                49: single_voxel_mask([5,9]),
                50: single_voxel_mask([6,8]),
                51: single_voxel_mask([5,8]),
                52: single_voxel_mask([5,8]),
                53: single_voxel_mask([4,7]),
                54: single_voxel_mask([5,9]),
                55: single_voxel_mask([5,8]),
                56: single_voxel_mask([5,9]),
            },
            "w3": {
                45: single_voxel_mask([5,8]),
                47: single_voxel_mask([6,6]),
                48: single_voxel_mask([7,6]),
                49: single_voxel_mask([6,7]),
                50: single_voxel_mask([4,8]),
                51: single_voxel_mask([5,8]),
                52: single_voxel_mask([7,7]),
                53: single_voxel_mask([5,8]),
                54: single_voxel_mask([6,8]),
                55: single_voxel_mask([5,8]),
                56: single_voxel_mask([6,7]),
            },
            "w6": {
                45: single_voxel_mask([5,8]),
                47: single_voxel_mask([4,7]),
                48: single_voxel_mask([4,8]),
                49: single_voxel_mask([4,7]),
                50: single_voxel_mask([5,8]),
                51: single_voxel_mask([5,8]),
                52: single_voxel_mask([7,6]),
                53: None,
                54: single_voxel_mask([6,8]),
                55: single_voxel_mask([5,8]),
                56: single_voxel_mask([5,9]),
            },
            "w9": {
                45: single_voxel_mask([6,8]),
                47: single_voxel_mask([5,8]),
                48: single_voxel_mask([6,7]),
                49: single_voxel_mask([5,7]),
                50: single_voxel_mask([6,8]),
                51: single_voxel_mask([5,7]),
                52: single_voxel_mask([7,7]),
                53: None,
                54: single_voxel_mask([7,7]),
                55: single_voxel_mask([4,8]),
                56: single_voxel_mask([5,8]),
            },
        },

        "Gastrocnemius": {
            "basal": {
                45: single_voxel_mask([10,10]),
                47: single_voxel_mask([11,9]),
                48: single_voxel_mask([11,12]),
                49: single_voxel_mask([11,12]),
                50: single_voxel_mask([9,12]),
                51: single_voxel_mask([9,13]),
                52: single_voxel_mask([10,11]),
                53: single_voxel_mask([12,11]),
                54: single_voxel_mask([11,10]),
                55: single_voxel_mask([5,13]),
                56: single_voxel_mask([10,12]),
            },
            "w3": {
                45: single_voxel_mask([10,13]),
                47: single_voxel_mask([5,14]),
                48: single_voxel_mask([8,14]),
                49: single_voxel_mask([5,13]),
                50: single_voxel_mask([11,12]),
                51: single_voxel_mask([11,10]),
                52: single_voxel_mask([11,10]),
                53: single_voxel_mask([8,14]),
                54: single_voxel_mask([7,13]),
                55: single_voxel_mask([10,12]),
                56: single_voxel_mask([10,12]),
            },
            "w6": {
                45: single_voxel_mask([10,12]),
                47: single_voxel_mask([11,12]),
                48: single_voxel_mask([8,14]),
                49: single_voxel_mask([10,13]),
                50: single_voxel_mask([10,11]),
                51: single_voxel_mask([4,13]),
                52: single_voxel_mask([8,13]),
                53: None,
                54: single_voxel_mask([9,14]),
                55: single_voxel_mask([11,11]),
                56: single_voxel_mask([10,12]),
            },
            "w9": {
                45: single_voxel_mask([11,10]),
                47: single_voxel_mask([11,10]),
                48: single_voxel_mask([5,13]),
                49: single_voxel_mask([7,13]),
                50: single_voxel_mask([5,13]),
                51: single_voxel_mask([9,13]),
                52: single_voxel_mask([12,10]),
                53: None,
                54: single_voxel_mask([11,11]),
                55: single_voxel_mask([7,14]),
                56: single_voxel_mask([7,13]),
            },
        },

        "Soleus": {
            "basal": {
                45: single_voxel_mask([8,9]),
                47: single_voxel_mask([8,11]),
                48: single_voxel_mask([8,10]),
                49: single_voxel_mask([8,11]),
                50: single_voxel_mask([8,11]),
                51: single_voxel_mask([8,11]),
                52: single_voxel_mask([8,11]),
                53: single_voxel_mask([5,12]),
                54: single_voxel_mask([9,10]),
                55: single_voxel_mask([8,10]),
                56: single_voxel_mask([8,11]),
            },
            "w3": {
                45: single_voxel_mask([8,11]),
                47: single_voxel_mask([6,11]),
                48: single_voxel_mask([5,11]),
                49: single_voxel_mask([5,12]),
                50: single_voxel_mask([8,11]),
                51: single_voxel_mask([8,10]),
                52: single_voxel_mask([8,11]),
                53: single_voxel_mask([9,10]),
                54: single_voxel_mask([9,10]),
                55: single_voxel_mask([9,10]),
                56: single_voxel_mask([9,10]),
            },
            "w6": {
                45: single_voxel_mask([8,11]),
                47: single_voxel_mask([9,11]),
                48: single_voxel_mask([10,12]),
                49: single_voxel_mask([9,12]),
                50: single_voxel_mask([8,10]),
                51: single_voxel_mask([7,10]),
                52: single_voxel_mask([5,11]),
                53: None,
                54: single_voxel_mask([8,10]),
                55: single_voxel_mask([7,12]),
                56: single_voxel_mask([9,11]),
            },
            "w9": {
                45: single_voxel_mask([9,10]),
                47: single_voxel_mask([10,11]),
                48: single_voxel_mask([8,11]),
                49: single_voxel_mask([10,10]),
                50: single_voxel_mask([7,12]),
                51: single_voxel_mask([9,12]),
                52: single_voxel_mask([4,10]),
                53: None,
                54: single_voxel_mask([6,11]),
                55: single_voxel_mask([5,10]),
                56: single_voxel_mask([6,10]),
            },
        },
    }
    return roi_masks
