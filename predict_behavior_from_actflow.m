function [output]=predict_behavior_from_actflow(behav_outcome,use_rest,activity,use_pca,zscore_features)

%Author: Ravi Mill, rdm146@rutgers.edu
%Last update: 22 June 2020

%DESCRIPTION: predicts individual differences in a behavioral outcome measure (Y) from
    %brain imaging features (X; task activations or restFC) via the leave-one-subject-out crossvalidated 
    %regression approach described in Mill, Gordon, Balota & Cole (2020): 
    %"Predicting dysfunctional age-related task activations from resting-state network alterations".

%INPUTS:
%behav_outcome: numeric vector, containing a to-be-predicted behavioral outcome measure (Y in the 
    %regression model) across subjects; dimensions=number of subjects,1
%use_rest: numeric binary, conveying number of tasks/states for input features; 
    %0=input features are task activations/vectorized FC from multiple
    %tasks (separate models are then built from each task's features, as well as using cross-task
    %features (1st principal component across tasks, estimated for each subject); 
    %1=features are task activations/FC from a single task/state (e.g. restFC)
%activity: numeric array, containing brain imaging features (X) that will be used to model behav_outcome;
    %dimensions vary according to number of tasks/states set in use_rest:
    %if use_rest=0 (multiple tasks), dimensions (3d)=number of activations/connections, number of tasks, number of subjects
    %if use_rest=1 (single task e.g. restFC), dimensions(2d)=number of activations/connections, number of subjects
%use_pca: numeric binary; 1=use PCA dimensionality reduction when training
    %the regression model (recommended if num_features > num_subs), number
    %of PCs retained i.e. ncomps will then equal the rank of the training
    %features (trainX); 
    %0=do not use PCA dimensionality reduction (recommended if num_features < num_subs)
%zscore_features: numeric, sets whether zscore normalization is applied during model training; 
    %0=no zscoring; 1=zscore features within each subject (this is noncircular); 2=zscore
    %features (X) across subjects (using mean/stdev from training set to prevent circularity), and 
    %zscore the behav outcome (Y; again using mean/stdev from the train set);
    %*note: zscore_features==2 is recommended for most applications.

%OUTPUTS:
%output: struct containing prediction accuracy for the trained model i.e.
    %overlap of the predicted and actual behavioral outcome measure
    %(Pearson r, and accompanying p value); note that the specific fields contained in output 
    %will depend on whether input features span multiple tasks/states, or
    %a single/task state (determined by 'use_rest'); 
    %e.g. if use_rest=0 (with two input task states), then the following outputs 
    %will be generated: task1_r, task1_p, task2_r, task2_p,
    %PC1_crosstask_r, PC1_crosstask_p

%recommended usage for regional activation (num regions > num subs): [output]=predict_behavior_from_actflow(behav_outcome,0,activity,1,2)
%recommended usage for network-averaged activation (num networks < num subs): [output]=predict_behavior_from_actflow(behav_outcome,0,activity,0,2)
%recommended usage for regional restFC (num connections > num subs): [output]=predict_behavior_from_actflow(behav_outcome,1,activity,1,2)
%recommended usage for network-averaged restFC (e.g. GBC; num networks < num subs): [output]=predict_behavior_from_actflow(behav_outcome,1,activity,0,2)

%extract info from inputs
num_subs=length(behav_outcome);
if use_rest==1
    num_tasks=1;
else
    num_tasks=size(activity,2);
end
num_features=size(activity,1);

%predict behavioral outcome from input features via leave-one-subject-out crossval
%select number of models to run based on use_rest (i.e. whether
%single/multiple task state features were input)
if use_rest==0
    %multiple tasks, including cross-task
    cv_loops=num_tasks+1;
elseif use_rest==1
    %single task only
    cv_loops=1;
end
    
for t=1:cv_loops
    %set output substruct name
    if t==num_tasks+1
        %predict from cross-task features
        t_name='PC1_crosstask';

        %compute cross-task features for each subject by taking 1st PC (latent variable approach)
        act=zeros(num_subs,num_features);
        for i=1:num_subs
            i_sub=activity(:,:,i);
            %demean prior to pca
            demean_trainX=repmat(mean(i_sub,1),size(i_sub,1),1);
            i_sub=i_sub-demean_trainX;
            %run pca and take 1st PC
            [PCALoadings,PCAScores,~,~,explained] = pca(i_sub);
            act(i,:)=PCAScores(:,1);
        end

    else
        t_name=['task',num2str(t)];
        %pull out activity for this specific task
        if num_tasks>1
            act=squeeze(activity(:,t,:))';
        else
            act=activity';
        end
    end

    %init behav variables
    actual_test_Y=[];
    pred_test_Y=[];
    for s=1:num_subs
        %set up LOSO crossval loop
        s_inds=1:num_subs;
        s_inds(s)=[];

        %set up train and test variables, factoring in zscore inputs
        if zscore_features==1
            %within subject zscore, applied to features (X) only
            t_act=zscore(act,0,2);
            train_X=[t_act(s_inds,:),ones(num_subs-1,1)];
            train_Y=behav_outcome(s_inds);

            test_X=t_act(s,:);
            test_Y=behav_outcome(s);
        elseif zscore_features==2
            %zscore features across subjects using training set mean/std, applied
            %to features (X) and behav outcomes (Y)
            X_mean=mean(act(s_inds,:),1);
            X_std=std(act(s_inds,:));
            Y_mean=mean(behav_outcome(s_inds));
            Y_std=std(behav_outcome(s_inds));

            %loop through X subtracting appropriate mean/std from each
            %element, applies this normalisation to both train and test features
            t_act=act;
            for rr=1:num_features
                t_act(:,rr)=(t_act(:,rr)-X_mean(rr))/X_std(rr);
            end
            %separate features into train/test (also add constant to trainX)
            train_X=[t_act(s_inds,:),ones(num_subs-1,1)];
            test_X=t_act(s,:);

            %zscore behav Y, and separate into train/test
            train_Y=(behav_outcome(s_inds)-Y_mean)/Y_std;
            test_Y=(behav_outcome(s)-Y_mean)/Y_std;

        elseif zscore_features==0
            %no zscoring applied
            t_act=act;
            train_X=[t_act(s_inds,:),ones(num_subs-1,1)];
            test_X=t_act(s,:);
            train_Y=behav_outcome(s_inds);
            test_Y=behav_outcome(s);
        end

        %train model, factoring in whether pca is applied to training features
        if use_pca==1
            %use pca regression
            train_X=train_X(:,1:(end-1)); %remove constant ones prior to pca

            %set max number of PCs, depends on num obs vs num feature vars
            if num_subs>num_features
                numComponents=num_features;
            else
                numComponents=min(size(train_X,1)-1,size(train_X,2)-1);
            end

            %demean prior to PCA
            demean_trainX=repmat(mean(train_X,1),size(train_X,1),1);
            train_X=train_X-demean_trainX;

            %run pca
            [PCALoadings,PCAScores,~,~,explained] = pca(train_X,'NumComponents',numComponents);
            
            %check for rank deficiency of PCAScores (resulting from rank deficiency of input train_X),
            %and if so adjust PCAScores accordingly
            PCA_rank=rank(PCAScores);
            if PCA_rank<numComponents
                PCAScores=PCAScores(:,1:PCA_rank);
                PCALoadings=PCALoadings(:,1:PCA_rank);
            end

            %regress after mean-centering trainY
            %*note that mean-centering train_Y has no effect if zscoring was done across subs (i.e. zscore_features=2))
            %also can ignore warning re lack of constant for PCAScores (given that PCAScores will be zero-mean)
            b1=regress(train_Y-mean(train_Y),PCAScores);

            %transform from PCA space back to original
            beta=PCALoadings*b1;

            %append constant to PCRbeta, see here for info https://stackoverflow.com/questions/11342759/constant-term-in-matlab-principal-component-regression-pcr-analysis
            %*note appending constant has no effect if applied to data that has been zscored across subs (i.e. zscore_features=2)
            beta=[beta;mean(train_Y)-mean(train_X)*beta]; 

        elseif use_pca==0

            %in cases of num obs < num features (or rank deficiency due to other reasons e.g. collinearity), this 'regress' call will 
            %use matlab's 'basic' regress solution (zero out redundant features using QR decomposition) which is a valid way 
            %to deal with rank deficiency, but one that will generate a warning
            %*note that mean-centering train_Y below has no effect if data was zscored across features (i.e. zscore_features=2)
            beta=regress(train_Y-mean(train_Y),train_X);
            
        end
        
        %apply trained model to test data
         %apply same transformation to test data i.e. center test_X according to demean_trainX 
        %*note this has no effect if test_X/Y have already been zscored across subs (i.e. zscore_features=2)
        demean_trainX=repmat(mean(train_X,1),size(train_X,1),1);
        if use_pca==0
            demean_trainX(:,end)=[]; %remove constant if present (only the case for non pca)
        end
        test_X=test_X-demean_trainX(1,:); 
        test_Y=test_Y-mean(train_Y);

        %compute predY based on trained model
        pred_Y=[test_X 1]*beta;

        %store pred_Y, and actual_Y
        pred_test_Y=[pred_test_Y;pred_Y];    
        actual_test_Y=[actual_test_Y;test_Y];

    end

    %compute model accuracy i.e. pred-to-actual behav overlap (i.e. yhat with y), and store
    [r,p]=corrcoef(actual_test_Y,pred_test_Y);
    output.([t_name,'_r'])=r(1,2);
    output.([t_name,'_p'])=p(1,2);

end

end

