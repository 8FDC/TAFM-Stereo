


def change_layer_name(name):
    """
    old_name --> new_name (list)
    """
    result = []

    if 'pretrained' in name:
        result.append(name.replace('pretrained', 'share_encoder'))
        result.append(name.replace('pretrained', 'dfm.encoder'))

    
    elif 'depth_head' in name:

        if ('projects' not in name) and ('resize_layers' not in name) and ('scratch.layer' not in name):
            # result.append(name.replace('depth_head', 'decoder'))
            result.append(name.replace('depth_head', 'dfm.decoder'))
            result.append(name.replace('depth_head', 'mono_decoder'))
        
        elif 'projects' in name:
            # result.append(name.replace('depth_head', 'readout_block'))
            result.append(name.replace('depth_head', 'dfm.readout_block'))

        
        elif 'resize_layers' in name:
            # result.append(name.replace('depth_head', 'readout_block'))
            result.append(name.replace('depth_head', 'dfm.readout_block'))
        
        elif 'scratch.layer' in name:
            # result.append(name.replace('depth_head', 'decoder'))
            result.append(name.replace('depth_head', 'dfm.decoder'))  # 冗余
            # result.append(name.replace('depth_head', 'scratch_head'))
            result.append(name.replace('depth_head', 'dfm.scratch_head'))  # 冗余

        
        else: raise ValueError
    
    else: raise ValueError

    return result


