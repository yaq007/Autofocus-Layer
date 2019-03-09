import sys
import normalizationModule
import os

testf = open('test.txt', 'r')
test_dir = testf.readlines()

for i in range(int(len(test_dir))):
    i = i + 1
    print "image" + str(i)
	
    if i % 5 == 4:
        continue

    direct_mask,_ = test_dir[5 * int(i/5)].split("\n")    
    direct_image,_ = test_dir[i].split("\n")    
    direct_mask = direct_mask + "/mask.nii.gz"
    direct_image = direct_image +".gz"
    
    pathToMainFolderWithSubjects = "./HGG/"
    subjectsToProcess = os.listdir(pathToMainFolderWithSubjects)
    subjectsToProcess.sort()
 
    saveOutput = True
    prefixToAddToOutp = "_zNorm2StdsMu"
     
    dtypeToSaveOutput = "float32"
    saveNormalizationPlots = True
         

    lowHighCutoffPercentile = [5., 95.]
    lowHighCutoffTimesTheStd = [3., 3.]
    cutoffAtWholeImgMean = True 
    normalizationModule.do_normalization( __file__,
				pathToMainFolderWithSubjects,
				subjectsToProcess,
				direct_image,direct_mask,				
				saveOutput,
				prefixToAddToOutp,				
				dtypeToSaveOutput,
				saveNormalizationPlots,								
				lowHighCutoffPercentile, # Can be None
				lowHighCutoffTimesTheStd, # Can be None
				cutoffAtWholeImgMean,
				)


