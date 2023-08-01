def convert_size(file_size):
    """
    """
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = 1024
    for i in range(len(units)):
        if (file_size / size) < 1:
            return "%.2f%s" % (file_size, units[i])
        file_size = file_size / size


def convert_model_type(model_type):
    if model_type == 'Lora':
        return 'lora'
    if model_type == 'checkpoints':
        return 'safetensors'
    
    return None