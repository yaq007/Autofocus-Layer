import SimpleITK as sitk
data_mha = open('data_mha.txt', 'r')
mha_dir = data_mha.readlines()
data_nii = open('data_nii.txt', 'r')
nii_dir = data_nii.readlines()

for i in range(len(mha_dir)):
    print(i)
    path, _ = mha_dir[i].split("\n")
    savepath, _ = nii_dir[i].split("\n")
    img = sitk.ReadImage(path)
    sitk.WriteImage(img, savepath)
