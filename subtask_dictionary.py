subtask_dict: dict[str, dict[str, dict[str, set[int] | list[list[int]]]]] = \
    {
        'beam': {
            'beam-0': {
                'room_set': {7},
                'subtask_goals': [[130, 252], [77, 134]]
            }, 'beam-1': {
                'room_set': {12},
                'subtask_goals': [[5, 235]]
            }, 'beam-2': {
                'room_set': {12},
                'subtask_goals': [[149, 235]]
            }, 'beam-3': {
                'room_set': {0},
                'subtask_goals': [[25, 252], [77, 134]]
            }
        },
        'traversal': {
            'traversal-0': {
                'room_set': {1},
                'subtask_goals': [[109, 200], [133, 148], [26, 148], [17, 207], [149, 235]]
            }, 'traversal-1': {
                'room_set': {14},
                'subtask_goals': [[68, 189], [85, 204], [77, 134]]
            }, 'traversal-2': {
                'room_set': {5},
                'subtask_goals': [[124, 200], [38, 78], [77, 252], [77, 134]]
            }, 'traversal-3': {
                'room_set': {5},
                'subtask_goals': [[124, 200], [38, 78], [77, 252], [77, 134]]
            }
        },
        'floor': {
            'floor-0': {
                'room_set': {20},
                'subtask_goals': [[77, 252], [140, 252], [5, 235]]
            }, 'floor-1': {
                'room_set': {18},
                'subtask_goals': [[5, 235]]
            }, 'floor-2': {
                'room_set': {22, 23},
                'subtask_goals': [[149, 235], [100, 252, 23], [115, 252], [130, 252], [5, 235], [5, 235, 22]]
            }
        },
        'jump': {
            'jump-0': {
                'room_set': {8},
                'subtask_goals': [[77, 252], [149, 253]]
            }
        },
        'monster': {
            'monster-0': {
                'room_set': {2, 6},
                'subtask_goals': [[77, 134, 2], [17, 252, 6]]
            }, 'monster-1': {
                'room_set': {9},
                'subtask_goals': [[5, 235]]
            }, 'monster-2': {
                'room_set': {3},
                'subtask_goals': [[77, 134]]
            }
        }
    }
