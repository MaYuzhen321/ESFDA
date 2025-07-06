from basic.split_dataset import all


if __name__ == '__main__':
    dataset = 'just'
    domain_path = r'G:\数据集\机械故障诊断数据集\JUST'
    save_path = r'G:\数据集\机械故障诊断数据集\JUST_USE'
    all(dataset=dataset, domain_path=domain_path, save_path=save_path)

