import os
import glob


#递归查找目录下指定后缀的文件
def find_file(path, suffix):
    result = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(suffix):
                result.append(os.path.join(root, file))
    return result

#search all files in the current directory
def search_file(suffix):
    result = []
    for file in glob.glob('*' + suffix):
        result.append(file)
    return result

#快速排序
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    left = [x for x in arr[1:] if x < pivot]
    middle = [x for x in arr[1:] if x == pivot]
    right = [x for x in arr[1:] if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

#5折交叉验证划分数据集
def cross_validation(data, k):
    data_size = len(data)
    fold_size = data_size // k
    for i in range(k):
        test = data[i * fold_size:(i + 1) * fold_size]
        train = data[:i * fold_size] + data[(i + 1) * fold_size:]
        yield train, test

#从excel中加载数据
def load_data(file_name):
    import xlrd
    data = xlrd.open_workbook(file_name)
    table = data.sheets()[0]
    nrows = table.nrows
    ncols = table.ncols
    result = []
    for i in range(nrows):
        row = table.row_values(i)
        result.append(row)
    return result

#使用pytorch的3D卷积神经网络    
def conv_net_3d(input_shape, output_shape):
    from torch.nn import Conv3d, MaxPool3d, Linear, ReLU, Dropout
    from torch.nn import Sequential
    model = Sequential(
        Conv3d(1, 32, 3, padding=1),
        ReLU(),
        MaxPool3d(2),
        Dropout(0.25),
        Conv3d(32, 64, 3, padding=1),
        ReLU(),
        MaxPool3d(2),
        Dropout(0.25),
        Conv3d(64, 128, 3, padding=1),
        ReLU(),
        MaxPool3d(2),
        Dropout(0.25),
        Linear(128, output_shape),
        ReLU()
    )
    return model

#按照列拼接两个二维数组
def concat_two_array_by_col(array1, array2):
    result = []
    for i in range(len(array1)):
        result.append(array1[i])
        result.append(array2[i])
    return np.array(result).T

#按照行拼接两个二维数组
def concat_two_array_by_row(array1, array2):
    result = []
    for i in range(len(array1)):
        result.append(array1[i])
        result.append(array2[i])
    return result

#查找指定数据的索引
def find_index(data, value):
    result = []
    for i in range(len(data)):
        if data[i] == value:
            result.append(i)
    return result

#递归从目录下删除指定文件夹
def remove_dir(path):
    if os.path.isdir(path):
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isdir(file_path):
                remove_dir(file_path)
            else:
                os.remove(file_path)
        os.rmdir(path)

#复制文件夹
def copy_dir(src, dst):
    if os.path.isdir(src):
        if not os.path.exists(dst):
            os.makedirs(dst)
        for file in os.listdir(src):
            file_path = os.path.join(src, file)
            if os.path.isdir(file_path):
                copy_dir(file_path, os.path.join(dst, file))
            else:
                shutil.copy(file_path, dst)

#选择排序
def select_sort(data):
    for i in range(len(data)):
        min_index = i
        for j in range(i + 1, len(data)):
            if data[j] < data[min_index]:
                min_index = j
        data[i], data[min_index] = data[min_index], data[i]
    return data

#加载pytorch模型权重
def load_model_weight(model, weight_path):
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path))
    else:
        print('weight file not exist')
#pytorch模型权重保存
def save_model_weight(model, weight_path):
    torch.save(model.state_dict(), weight_path)

#计算两个数组的相似度
def cal_similarity(array1, array2):
    result = 0
    for i in range(len(array1)):
        result += array1[i] * array2[i]
    return result

#使用交叉熵损失函数
def cross_entropy_loss(output, target):
    return -torch.mean(target * torch.log(output + 1e-8))
#从二分类预测概率中得到分类结果
def get_class_from_prob(prob):
    result = []
    for i in range(len(prob)):
        if prob[i] > 0.5:
            result.append(1)
        else:
            result.append(0)
    return result
#从多分类预测中得到分类结果
def get_class_from_prob_multi(prob):
    result = []
    for i in range(len(prob)):
        max_index = 0
        max_value = 0
        for j in range(len(prob[i])):
            if prob[i][j] > max_value:
                max_value = prob[i][j]
                max_index = j
        result.append(max_index)
    return result

#统计每个文件夹内的文件数量
def count_file_num(path):
    result = {}
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            result[file] = count_file_num(file_path)
        else:
            if file not in result:
                result[file] = 1
            else:
                result[file] += 1
    return result

#保存到excel
def save_to_excel(data, path):
    if os.path.exists(path):
        os.remove(path)
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet('sheet1')
    for i in range(len(data)):
        for j in range(len(data[i])):
            sheet.write(i, j, data[i][j])
    workbook.save(path)
#从excel中读取数据
def read_from_excel(path):
    data = []
    if os.path.exists(path):
        workbook = xlrd.open_workbook(path)
        sheet = workbook.sheet_by_index(0)
        for i in range(sheet.nrows):
            data.append(sheet.row_values(i))
    return data
#从excel中读取数据，并转换为数组
def read_from_excel_to_array(path):
    data = []
    if os.path.exists(path):
        workbook = xlrd.open_workbook(path)
        sheet = workbook.sheet_by_index(0)
        for i in range(sheet.nrows):
            data.append(sheet.row_values(i))
    return np.array(data)
#numpy与list转换
def numpy_to_list(data):
    result = []
    for i in range(len(data)):
        result.append(data[i].tolist())
    return result
#删除所有文件名中的空格
def remove_space_from_file_name(path):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            remove_space_from_file_name(file_path)
        else:
            os.rename(file_path, file_path.replace(' ', ''))

#pytorch数据增强
def data_augmentation(data, label, batch_size, transform):
    data_num = len(data)
    data_index = np.arange(data_num)
    np.random.shuffle(data_index)
    data_index = data_index.tolist()
    for i in range(0, data_num, batch_size):
        batch_data = []
        batch_label = []
        for j in range(batch_size):
            if i + j < data_num:
                batch_data.append(data[data_index[i + j]])
                batch_label.append(label[data_index[i + j]])
        yield transform(torch.Tensor(batch_data)), torch.Tensor(batch_label)

#区域增长3D分割
def region_growing_3d(data, label, seed_point, threshold, max_iter):
    data_num = len(data)
    data_index = np.arange(data_num)
    np.random.shuffle(data_index)
    data_index = data_index.tolist()
    seed_point = seed_point.tolist()
    seed_point_num = len(seed_point)
    seed_point_index = np.arange(seed_point_num)
    np.random.shuffle(seed_point_index)
    seed_point_index = seed_point_index.tolist()
    for i in range(max_iter):
        for j in range(seed_point_num):
            seed_point_index[j] = data_index[seed_point[seed_point_index[j]]]
        for j in range(seed_point_num):
            seed_point[j] = data_index[seed_point_index[j]]
        for j in range(seed_point_num):
            seed_point_index[j] = seed_point[seed_point_index[j]]
        for j in range(seed_point_num):
            seed_point_index[j] = seed_point[seed_point_index[j]]
        for j in range(seed_point_num):
            seed_point_index[j] = seed_point[seed_point_index[j]]

#计算质数
def get_prime_number(num):
    result = []
    for i in range(2, num + 1):
        if is_prime(i):
            result.append(i)
    return result
#判断是否为质数
def is_prime(num):
    if num == 1:
        return False
    for i in range(2, int(math.sqrt(num)) + 1):
        if num % i == 0:
            return False
    return True
#数组保存为图片
def save_array_to_image(data, path):
    if os.path.exists(path):
        os.remove(path)
    data = np.array(data)
    data = data.astype(np.uint8)
    data = data.reshape(data.shape[0], data.shape[1], 1)
    data = np.concatenate((data, data, data), axis=2)
    image = Image.fromarray(data)
    image.save(path)
#hwc转为chw
def hwc_to_chw(data):
    data = np.array(data)
    data = np.transpose(data, (2, 0, 1))
    return data
#搜索网页中的图片
def search_image_in_web(url, path):
    if os.path.exists(path):
        os.remove(path)
    html = requests.get(url).text
    soup = BeautifulSoup(html, 'lxml')
    img_url = soup.find_all('img')
    for i in range(len(img_url)):
        img_url[i] = img_url[i].get('src')
    for i in range(len(img_url)):
        img_url[i] = img_url[i].replace('\\', '')
    img_url = list(set(img_url))
    for i in range(len(img_url)):
        img_url[i] = img_url[i].replace(' ', '')
    img_url = list(set(img_url))
    for i in range(len(img_url)):
        img_url[i] = img_url[i].replace('\n', '')
    img_url = list(set(img_url))
    for i in range(len(img_url)):
        img_url[i] = img_url[i].replace('\t', '')
    img_url = list(set(img_url))
    for i in range(len(img_url)):
        img_url[i] = img_url[i].replace('\r', '')
    img_url = list(set(img_url))
    for i in range(len(img_url)):
        img_url[i] = img_url[i].replace('\b', '')
    img_url = list(set(img_url))
    for i in range(len(img_url)):
        img_url[i] = img_url[i].replace('\f', '')
    img_url = list(set(img_url))
    for i in range(len(img_url)):
        img_url[i] = img_url[i].replace('\v', '')
    img_url = list(set(img_url))
    for i in range(len(img_

#保存网页中的图片
def save_image_in_web(url, path):
    if os.path.exists(path):
        os.remove(path)
    html = requests.get(url).text
    soup = BeautifulSoup(html, 'lxml')
    img_url = soup.find_all('img')
    for i in range(len(img_url)):
        img_url[i] = img_url[i].get('src')
    for i in range(len(img_url)):
        img_url[i] = img_url[i].replace('\\', '')
    img_url = list(set(img_url))
    for i in range(len(img_url)):
        img_url[i] = img_url[i].replace(' ', '')
    img_url = list(set(img_url))
    for i in range(len(img_url)):
        img_url[i] = img_url[i].replace('\n', '')
    img_url = list(set(img_url))
    for i in range(len(img_url)):
        img_url[i] = img_url[i].replace('\t', '')
    img_url = list(set(img_url))
    for i in range(len(img_url)):
        img_url[i] = img_url[i].replace('\r', '')

#绘制roc曲线
def draw_roc_curve(y_true, y_score, path):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(path)
    plt.show()        

#自动单击网页中的按钮
def auto_click_button(url, button_name):
    html = requests.get(url).text
    soup = BeautifulSoup(html, 'lxml')
    button = soup.find_all(button_name)
    for i in range(len(button)):
        button[i] = button[i].get('onclick')
    for i in range(len(button)):
        button[i] = button[i].replace(' ', '')
    button = list(set(button))
    for i in range(len(button)):
        button[i] = button[i].replace('\n', '')
    button = list(set(button))
    for i in range(len(button)):
        button[i] = button[i].replace('\t', '')
    button = list(set(button))

#分类指标
def classification_index(y_true, y_score):
    precision = precision_score(y_true, y_score)
    recall = recall_score(y_true, y_score)
    f1 = f1_score(y_true, y_score)
    print('precision:', precision)
    print('recall:', recall)
    print('f1:', f1)

#计算混淆矩阵
def confusion_matrix(y_true, y_score):
    cm = confusion_matrix(y_true, y_score)
    print(cm)

#计算分类报告
def classification_report(y_true, y_score):
    report = classification_report(y_true, y_score)
    print(report)

#遍历文件夹
def traverse_folder(path):
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list