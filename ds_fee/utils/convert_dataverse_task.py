import csv

def convert_to_csv(input_file_path, output_file_path):
    # 读取输入文件内容
    with open(input_file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 通过空行分割不同的块
    blocks = content.split('\n\n')
    # 去除每个块前后的空白字符
    blocks = [block.strip() for block in blocks if block.strip()]

    # 第一个块是CSV的文件头
    headers = [line.strip() for line in blocks[0].split('\n') if line.strip()]

    data_rows = []
    for block in blocks[1:]:
        lines = block.split('\n')
        line_index = 0
        row = []

        # 第一行对应第一列数据
        if line_index < len(lines):
            row.append(lines[line_index].strip())
            line_index += 1

        # 第二行忽略
        if line_index < len(lines):
            line_index += 1

        # 第3 - 6行分别对应CSV中第2 - 5列的数据
        for _ in range(4):
            if line_index < len(lines):
                row.append(lines[line_index].strip())
                line_index += 1

        # 第7行忽略
        if line_index < len(lines):
            line_index += 1

        # 第8行对应第6列数据
        if line_index < len(lines):
            row.append(lines[line_index].strip())
            line_index += 1

        tmp = 0
        while tmp < 3:  # 最后一行是几个点，忽略
            line = lines[line_index].strip()
            if line:
                row.extend([part for part in line.split() if part])
            line_index += 1
            tmp += 1
        row.append("-")
        data_rows.append(row)

    # 写入CSV文件
    with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # 写入文件头
        writer.writerow(headers)
        # 写入数据行
        writer.writerows(data_rows)

# 使用示例
if  __name__ == '__main__':
    input_file = '/Users/liuhua2/Documents/gpt-dataverse-task.txt'
    output_file = '/Users/liuhua2/Documents/gpt-dataverse-task.csv'
    convert_to_csv(input_file, output_file)