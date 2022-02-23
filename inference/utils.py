


def get_inference(args):
    if args.dimension == '2d':
        if args.sliding_window:
            from .inference2d import inference_sliding_window
            return inference_sliding_window
        else:
            from .inference2d import inference_whole_image
            return inference_whole_image

    elif args.dimension == '3d':
        if args.sliding_window:
            from .inference3d import inference_sliding_window
            return inference_sliding_window

        else:
            from .inference3d import inference_whole_image
            return inference_whole_image
        
    

    else:
        raise ValueError('Error in image dimension')



def split_idx(half_win, size, i):
    '''
    half_win: The size of half window
    size: img size along one axis
    i: the patch index
    '''

    start_idx = half_win * i
    end_idx = start_idx + half_win*2

    if end_idx > size:
        start_idx = size - half_win*2
        end_idx = size

    return start_idx, end_idx

