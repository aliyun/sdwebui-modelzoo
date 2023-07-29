def convert_size(file_size):
    """
    """
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = 1024
    for i in range(len(units)):
        if (file_size / size) < 1:
            return "%.2f%s" % (file_size, units[i])
        file_size = file_size / size
