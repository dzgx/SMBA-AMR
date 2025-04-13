
def get_classes(from_file="./classes.txt"):

    classes_file = open(from_file)
    classes = []
    # strip()方法去除字符串左右两侧的空格和特殊字符
    for line in classes_file.readlines():
        classes.append(line.strip())
    return classes


if __name__ == '__main__':
    get_classes(from_file='./classes.txt')
