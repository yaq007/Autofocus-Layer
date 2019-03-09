import numpy as np
import os
import subprocess
import sys
import nibabel as nib

def saveImageToANewNiiWithHeaderFromOther(	normalizedImageNpArray,
                                          	outputFilepath,
                                          	originalImageProxy,
						dtypeToSaveOutput) : #the proxy

    hdr_for_orig_image = originalImageProxy.header
    affine_trans_to_ras = originalImageProxy.affine
    
    #Nifti Constructor. data is the image itself, dimensions x,y,z,time. The second argument is the affine RAS transf.
    newNormImg = nib.Nifti1Image(normalizedImageNpArray, affine_trans_to_ras) 
    newNormImg.set_data_dtype(np.dtype(dtypeToSaveOutput))
    newNormImg.header.set_zooms(hdr_for_orig_image.get_zooms()[:3])
    
    print("Saving image to:"),outputFilepath
    nib.save(newNormImg, outputFilepath)
    print("Done.")

#simple subtract mean divide std.
def do_normalization(windowTitle,
			pathToMainFolderWithSubjects,
			subjectsToProcess,
			channelToNormalizeFilepath, roiFilepath,			
			saveOutput,
			prefixToAddToOutp,			
			dtypeToSaveOutput,
			saveNormalizationPlots,						
			lowHighCutoffPercentile, # Can be None
			lowHighCutoffTimesTheStd, # Can be None
			cutoffAtWholeImgMean,
			) :
	for i in range(1) :
		srcProxy = nib.load(channelToNormalizeFilepath)
		srcImgNpArr = np.asarray(srcProxy.get_data(), dtype=dtypeToSaveOutput)
		boolPixelsWithIntBelowZero = srcImgNpArr<0
		numVoxelsIntBelowZero = np.sum(boolPixelsWithIntBelowZero)
		if numVoxelsIntBelowZero != 0:
			srcImgNpArr[boolPixelsWithIntBelowZero] = 0; #because for instance in Christian's  background was -1.
		meanIntSrcImg = np.mean(srcImgNpArr); stdIntSrcImg = np.std(srcImgNpArr); maxIntSrcImg = np.max(srcImgNpArr)       
	roiProxy = nib.load(roiFilepath)
	roiNpArr = np.asarray(roiProxy.get_data()) 
	boolRoiMask = roiNpArr>0
	
	srcIntsUnderRoi = srcImgNpArr[boolRoiMask] # This gets flattened automatically. It's a vector array.
	meanIntInRoi = np.mean(srcIntsUnderRoi); stdIntInRoi = np.std(srcIntsUnderRoi); maxIntInRoi = np.max(srcIntsUnderRoi)
	print("\t(in ROI) Intensity Mean:", meanIntInRoi, ", Std :", stdIntInRoi, ", Max: "), maxIntInRoi
	
	### Normalize ###
	print("\t...Normalizing...")
	boolMaskForStatsCalc = boolRoiMask
	meanForNorm = meanIntInRoi
	stdForNorm = stdIntInRoi
	print("\t\t*Stats for normalization changed to: Mean=", meanForNorm, ", Std="), stdForNorm
	if lowHighCutoffPercentile is not None and lowHighCutoffPercentile != [] :
		lowCutoff = np.percentile(srcIntsUnderRoi, lowHighCutoffPercentile[0]);		
		boolOverLowCutoff = srcImgNpArr > lowCutoff
		highCutoff = np.percentile(srcIntsUnderRoi, lowHighCutoffPercentile[1]);	
		boolBelowHighCutoff = srcImgNpArr < highCutoff
		print("\t\tCutting off intensities with [percentiles] (within Roi). Cutoffs: Min=", lowCutoff ,", High="), highCutoff
		boolMaskForStatsCalc = boolMaskForStatsCalc * boolOverLowCutoff * boolBelowHighCutoff
		meanForNorm = np.mean(srcImgNpArr[boolMaskForStatsCalc])
		stdForNorm = np.std(srcImgNpArr[boolMaskForStatsCalc])
		print("\t\t*Stats for normalization changed to: Mean=", meanForNorm, ", Std="), stdForNorm

	if lowHighCutoffTimesTheStd is not None and lowHighCutoffTimesTheStd != [] :
		lowCutoff = meanIntInRoi - lowHighCutoffTimesTheStd[0] * stdIntInRoi;		
		boolOverLowCutoff = srcImgNpArr > lowCutoff
		highCutoff = meanIntInRoi + lowHighCutoffTimesTheStd[1] * stdIntInRoi;		
		boolBelowHighCutoff = srcImgNpArr < highCutoff
		print("\t\tCutting off intensities with [std] (within Roi). Cutoffs: Min=", lowCutoff ,", High="), highCutoff

		# The next 2 lines are for monitoring only. Could be deleted
		boolInRoiAndWithinCutoff = boolRoiMask * boolOverLowCutoff * boolBelowHighCutoff
		print("\t\t(In Roi, within THIS cutoff) Intensities Mean="), np.mean(srcImgNpArr[boolInRoiAndWithinCutoff]),", Std=", np.std(srcImgNpArr[boolInRoiAndWithinCutoff])
		
		boolMaskForStatsCalc = boolMaskForStatsCalc * boolOverLowCutoff * boolBelowHighCutoff
		meanForNorm = np.mean(srcImgNpArr[boolMaskForStatsCalc])
		stdForNorm = np.std(srcImgNpArr[boolMaskForStatsCalc])
		print("\t\t*Stats for normalization changed to: Mean=", meanForNorm, ", Std="), stdForNorm
	if cutoffAtWholeImgMean :
		lowCutoff = meanIntSrcImg;	boolOverLowCutoff = srcImgNpArr > lowCutoff
		print("\t\tCutting off intensities with [below wholeImageMean for air]. Cutoff: Min="), lowCutoff
		boolMaskForStatsCalc = boolMaskForStatsCalc * boolOverLowCutoff
		meanForNorm = np.mean(srcImgNpArr[boolMaskForStatsCalc])
		stdForNorm = np.std(srcImgNpArr[boolMaskForStatsCalc])
		print("\t\t*Stats for normalization changed to: Mean=", meanForNorm, ", Std="), stdForNorm
	
	    # Apply the normalization
		normImgNpArr = srcImgNpArr - meanForNorm
		normImgNpArr = normImgNpArr / (1.0*stdForNorm)
		print("\tImage was normalized using: Mean=", meanForNorm, ", Std="), stdForNorm
	
        ### Save output ###
	if saveOutput:
		outputFilepath = channelToNormalizeFilepath
		print(outputFilepath)
		print("\tSaving normalized output to: "),outputFilepath
		saveImageToANewNiiWithHeaderFromOther(normImgNpArr, outputFilepath, srcProxy, dtypeToSaveOutput)
	
        
