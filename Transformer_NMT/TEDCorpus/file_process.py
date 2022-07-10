import random
size = 402918
LA = "cn"
with open("TED2020.en-zh_cn.zh_cn", mode="r", encoding="utf-8") as f:
    train = int(size * 0.7)
    test = int(size * 0.9)
    print("size: ", size)
    print("train:", train)
    print("test: ", test)
    with open(f"train_{LA}.txt","w",encoding="utf-8") as en_train:
        with open(f"dev_{LA}.txt", "w",encoding="utf-8") as en_dev:
            with open(f"test_{LA}.txt", "w",encoding="utf-8") as en_test:
                a = 0
                for i, line in enumerate(f.readlines()):
                    # print(line)
                    # print(i)
                    if i < train:
                        en_train.write(line)
                    if train <= i < test:
                        en_dev.write(line)
                    if test <= i:
                        en_test.write(line)
                en_test.close()
                en_dev.close()
                en_train.close()
                f.close()
                exit(1)


