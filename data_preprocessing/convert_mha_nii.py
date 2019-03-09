import SimpleITK as sitk
trainf = open('testmha.txt', 'r')
train_dir = trainf.readlines()
testf = open('test2015.txt', 'r')
test_dir = testf.readlines()

for i in range(len(train_dir)):
    print(i)
    path, _ = train_dir[i].split("\n")
    savepath, _ = test_dir[i].split("\n")
    img = sitk.ReadImage(path)
    sitk.WriteImage(img, savepath)
