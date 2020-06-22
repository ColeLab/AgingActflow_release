function [output]=compute_multregFC_optPCA(activityMatrix, outputfile, ncomps_pre)

%Author: Ravi Mill, rdm146@rutgers.edu
%Last update: 22 June 2020

%DESCRIPTION: computes functional connectivity (FC) via the PCA-optimized 
    %multiple regression approach described in Mill, Gordon, Balota & Cole (2020): 
    %"Predicting dysfunctional age-related task activations from resting-state network alterations".
    %Briefly, number of PCs (ncomps) retained in each source region timeseries (X) -> target region (Y) timeseries 
    %multiple linear regression is optimized, by selecting the ncomps value (from 1:max ncomps permitted by the 
    %data) that yields the max cross-session FC similarity, AND also accounts
    %for at least 50% PCA variance explained. 
    
 %INPUTS:
 %activityMatrix: 2d numeric array containing activation timeseries from
    %which FC is estimated; input should be concatenated across sessions i.e. 
    %the first half of available timepoints is treated as session 1, and the second half of timepoints is 
    %treated as session 2; dimensions=number of regions/voxels,number of timepoints
 %outputfile: string, full path to where 'output' struct will be saved
 %ncomps_pre: numeric; default=empty to run full optimization procedure which determines the optimal 
    %number of PCs (ncomps) retained in the FC regressions for each subject; if set to a numeric value, 
    %this will confine the number of PCs retained to this pre-defined value 
    %(without going through the optimization)
 
 %OUTPUTS:
 %output: struct containing all output variables from the function, with the following fields:
 %netMat_final: numeric array, containing final FC estimates resulting from multiple
    %regressions over sources/targets that retained the optimized ncomps (or ncomps_pre), as 
    %estimated from the timeseries concatenated across sessions; note that this FC matrix is 
    %asymmetric, with columns corresponding to FC estimates for each target (i.e. source(X)->target(Y), 
    %the appropriate orientation for activity flow modeling); dimensions=number of regions/voxels,
    %number of regions/voxels
 %FCsimilarity: numeric array, containing the FCsimilarity (Pearson
    %correlation of the off-diagonal FC matrix elements across session1 and
    %session2) for each ncomps value; dimensions:rows=ncomps,cols(2):col1=FC
    %similarity r value, col2=corresponding p value
 %PCexplained: numeric array, containing variance explained by the PCA of
    %a set of source timeseries corresponding to a given target, averaged
    %across all targets at each optimization step (i.e. PCA variance explained across 
    %all regressions conducted for each ncomps value); 
    %dimensions: rows=ncomps, cols(3):col1=PCA explained for session1,
    %col2=PCA explained for session2,col3=PCA explained averaged across
    %session1-2 (col3 was reported as the overall PCexplained in the original paper)
 %REGexplained: numeric array, containing variance explained (R squared) by the
    %regression of a set of source timeseries predicting a given target, averaged
    %across all targets at each optimization step (i.e. across all regressions conducted 
    %for each ncomps value); dimensions:rows=ncomps, cols(3):col1=regression variance 
    %explained for session1,col2=variance explained for session2,col3=variance explained averaged 
    %across session1-2
 %opt_comps_info: numeric vector, containing final summary of optimal ncomps for this subject;
    %dimensions:rows=1,cols(4):col1=optimal ncomps value, col2=corresponding FCsimilarity r value, 
    %col3=corresponding PCexplained (cross-session), col4=corresponding REGexplained (cross-session)
 %opt_PCexplained: numeric, PCexplained for the final multiple regression
    %FC (i.e. for netMat_final, estimated from the concatenated session timeseries, 
    %after selecting an optimal ncomps value), averaged across all targets
 %opt_REGexplained: numeric, REGexplained for the final multiple regression
    %FC (i.e. for netMat_final), averaged across all targets


%% 1. Split timeseries in half
timeseries_length=size(activityMatrix,2);
split1=activityMatrix(:,1:(timeseries_length/2));
split2=activityMatrix(:,(timeseries_length/2)+1:timeseries_length);

%demean timeseries for each region
split1=split1-repmat(mean(split1,2),1,size(split1,2));
split2=split2-repmat(mean(split2,2),1,size(split2,2));

%find max number of components
maxComponents=min(size(split1,1)-1,size(split1,2)-1);
numRegions=size(activityMatrix,1);

%set ncomps_pre if it hasn't been set in the function call
if ~exist('ncomps_pre','var')
    ncomps_pre=[];
end

if isempty(ncomps_pre)
    %perform optimization by looping through ncomps=1:maxComponents
    
    %init outputs
    FCsimilarity=[]; %Pearson r correlation across split1 and split2; 2nd col = p value
    PCexplained=[]; %mean variance (across targetRegions) explained by the PCA for each region (separate cols for split1 and split2, mean in third col)
    REGexplained=[]; %mean variance (across targetRegions) explained by the regression i.e. the Rsquared result after the multiple linear regression (separate cols for split1 and split2, mean in third col)
    for ncomps=1:maxComponents
        
        %***compute FC for split1***
        netMat_split1=zeros(numRegions,numRegions);
        PCex=[]; %total PCA variance across targets
        REGex=[]; %total regression R2 across targets
        disp(['Computing optimization, ncomps= ',num2str(ncomps)]);
        for targetRegion=1:numRegions
            otherRegions=1:numRegions;
            otherRegions(targetRegion)=[];

            %Run PCA
            [PCALoadings,PCAScores,~,~,explained] = pca(split1(otherRegions,:)','NumComponents',ncomps);
            
            %add PCA variance explained to PCex
            PCex=[PCex;sum(explained(1:ncomps))]; %take the sum across all components

            %pull out target timeseries
            targetTS=split1(targetRegion,:);
            
            %conduct regression with PCAScores(X) predicting targetTS (Y)
            %*note constant is added to avoid matlab regress warnings, but
            %it is technically unnecessary given that each source region's timeseries
            %was demeaned above
            [pcabetas,~,~,~,stats] = regress(targetTS',[PCAScores(:,1:ncomps),ones(size(PCAScores,1),1)]);
            
            %store Rsquared
            REGex=[REGex;stats(1)];

            %transform regression betas from PCA space back to original space
            betaPCR = PCALoadings*pcabetas(1:length(pcabetas)-1);
            
            %store FC estimates for this targetRegion
            netMat_split1(otherRegions,targetRegion)=betaPCR;

        end
        
        %take the mean PCex and REGex across target region regressions
        s1_PCex=mean(PCex);
        s1_REGex=mean(REGex);

        %***compute FC for split2***
        netMat_split2=zeros(numRegions,numRegions);
        PCex=[]; %total PCA variance
        REGex=[]; %total regression R2
        for targetRegion=1:numRegions
            otherRegions=1:numRegions;
            otherRegions(targetRegion)=[];

            %Run PCA
            [PCALoadings,PCAScores,~,~,explained] = pca(split2(otherRegions,:)','NumComponents',ncomps);
            
            %add PCA variance explained to PCex
            PCex=[PCex;sum(explained(1:ncomps))];

            %pull out target timeseries
            targetTS=split2(targetRegion,:);
            
            %conduct regression with PCAScores(X) predicting targetTS (Y)
            %*note constant is added to avoid matlab regress warnings, but
            %it is technically unnecessary given that each source region's timeseries
            %was demeaned above
            [pcabetas,~,~,~,stats] = regress(targetTS',[PCAScores(:,1:ncomps),ones(size(PCAScores,1),1)]);
            
            %store Rsquared
            REGex=[REGex;stats(1)];

            %transform regression betas from PCA space back to original space
            betaPCR = PCALoadings*pcabetas(1:length(pcabetas)-1);
            
            %store FC estimates for this targetRegion
            netMat_split2(otherRegions,targetRegion)=betaPCR;

        end
        
        %take the mean PCex and REGex across target region regressions
        s2_PCex=mean(PCex);
        s2_REGex=mean(REGex);

        %concat PCexplained and REGexplained across session1-2
        ncomp_PCex=[s1_PCex,s2_PCex,mean([s1_PCex,s2_PCex])];
        PCexplained=[PCexplained;ncomp_PCex];

        ncomp_REGex=[s1_REGex,s2_REGex,mean([s1_REGex,s2_REGex])];
        REGexplained=[REGexplained;ncomp_REGex];

        %***Compute FCsimilarity***
        % Find logical index to the offdiagonal elements
        N=size(netMat_split1,1);
        [ii,jj]=ind2sub([N,N],1:N^2);
        notDiagonal=ii~=jj;
        
        % Use the logical index to vectorize the FCmatrix whilst excluding the
        % on-diagonal elements
        offDiagonalA=netMat_split1(notDiagonal);
        offDiagonalB=netMat_split2(notDiagonal);
        
        % Calculate the correlation (similarity) between these elements
        [r,p] = corrcoef(offDiagonalA,offDiagonalB);
        corrval=[r(1,2),p(1,2)];
        
        % Store
        FCsimilarity=[FCsimilarity;corrval];

    end

    %% 4. Choose optimal ncomps (based on max FCsimilarity)
    
    %identify optimal ncomps based on maximum FCsimilarity, AND 
    %at least 50% PCA variance explained
    pc50=0;
    while pc50==0
        [max_FCsim,comp_ind]=max(FCsimilarity(:,1));
        if PCexplained(comp_ind)>=50
            pc50=1;
        else
            %replace FCsimilarity with nan, so that next loop ignores this ncomps
            FCsimilarity(comp_ind)=NaN;
        end  
    end
    opt_comps_info=[comp_ind,max_FCsim,PCexplained(comp_ind,3),REGexplained(comp_ind,3)];
else
    comp_ind=ncomps_pre;
end

%% 5. Apply optimal ncomps to whole timeseries to derive final FCmatrix

%set optimized number of components
ncomps=comp_ind; 

%initialize final FC matrix
netMat_final=zeros(numRegions,numRegions);

%demean timeseries for each region
activityMatrix=activityMatrix-repmat(mean(activityMatrix,2),1,size(activityMatrix,2));

%init variance explained terms for final FC
PCex=[]; %total PCA variance
REGex=[]; %total regression variance (Rsquared)

%loop through target regions and compute FC
for targetRegion=1:numRegions
    otherRegions=1:numRegions;
    otherRegions(targetRegion)=[];
    
    %Run PCA
    [PCALoadings,PCAScores,~,~,explained] = pca(activityMatrix(otherRegions,:)','NumComponents',ncomps);
    
    %add total variance explained to PCex
    PCex=[PCex;sum(explained(1:ncomps))]; %take the sum across all components

    %pull out target timeseries
    targetTS=activityMatrix(targetRegion,:);

    %conduct regression with PCAScores(X) predicting targetTS (Y)
    %*note constant is added to avoid matlab regress warnings, but
    %it is technically unnecessary given that each source region's timeseries
    %was demeaned above
    [pcabetas,~,~,~,stats] = regress(targetTS',[PCAScores(:,1:ncomps),ones(size(PCAScores,1),1)]);
    
    %store Rsquared
    REGex=[REGex;stats(1)];

    %transform regression betas from PCA space back to original space
    betaPCR=PCALoadings*pcabetas(1:length(pcabetas)-1);

    %store FC estimates for this targetRegion
    netMat_final(otherRegions,targetRegion)=betaPCR;

end

%average variance explained terms over targets
opt_PCexplained=mean(PCex);
opt_REGexplained=mean(REGex);

%% 6. Assign output and save

if isempty(ncomps_pre)
    %assign to output
    output.netMat_final=netMat_final;
    output.FCsimilarity=FCsimilarity;
    output.PCexplained=PCexplained;
    output.REGexplained=REGexplained;
    output.opt_comps_info=opt_comps_info;
    output.opt_PCexplained=opt_PCexplained;
    output.opt_REGexplained=opt_REGexplained;
    
    
else
    %assign to output; exclude optimization results/metrics if using ncomps_pre
    output.netMat_final=netMat_final;
    output.opt_PCexplained=opt_PCexplained;
    output.opt_REGexplained=opt_REGexplained;
    
end

%save
save(outputfile,'output');


end
