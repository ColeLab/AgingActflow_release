function [output] = actflowmapping_aging(taskActMatrix_predSub, connMatrix_predSub, taskActMatrix_group, run_dysfunc_reg)

%Author: Ravi Mill, rdm146@rutgers.edu
%Last update: 22 June 2020

%DESCRIPTION: generates task activation predictions via the activity flow mapping procedure 
    %described in Mill, Gordon, Balota & Cole (2020): "Predicting dysfunctional 
     %age-related task activations from resting-state network alterations"

%INPUTS
%taskActMatrix_predSub: 3d numeric array with task activations for to-be-predicted 'unhealthy' subjects 
    %(e.g. at-risk AD subjects); dimensions=number of regions/voxels,number of tasks,number of subjects
%connMatrix_predSub: 4d numeric array with connectivity estimates (FC) for the
    %same to-be-predicted unhealthy subjects; dimensions=number of regions/voxels,
    %number of regions/voxels,number of tasks/states (e.g. 1 for restFC),number of subjects
%taskActMatrix_group: 2d numeric array with task activation templates for the
    %'healthy' group; dimensions=number of regions/voxels, number of tasks
%run_dysfunc_reg: numeric binary (default=0); 1=confines task activation prediction to dysfunctional activations
    %by regressing out the group healthy activations (taskActMatrix_group)
    %from the predicted and actual unhealthy activations prior to computing
    %their overlap (i.e. Pearson r prediction accuracy)

%OUTPUTS
%output: struct containing all output variables from the function, with the following fields:
%subject-level prediction accuracy: r_overall (prediction accuracy across tasks 
    %computed for each subject, then averaged across subjects), p_overall (accompanying p value), t_overall (accompanying t value); 
    %r_bytask (subject prediction accuracy computed for each task separately), p_bytask, t_bytask; 
    %r_bysubj (r value for each individual task and subject, dimensions=tasks, subjects), p_bysubj
%actual and predicted task activation arrays for unhealthy subjects: taskActualMatrix, taskPredMatrix
%group-level prediction accuracy: r_avgfirst_mean (group prediction accuracy across tasks computed after averaging
    %predicted and actual task activations across subjects and tasks), p_avgfirst_mean (accompanying p value); 
    %r_avgfirst_bytask (group prediction accuracy computed for each task separately), p_avgfirst_bytask

%make sure inputs are formatted correctly
if ndims(connMatrix_predSub)==3
    msg='The connectivity matrix should be 4 dimensions (regions X regions X states X subjects). An empty third dimension is being added (assuming you have only a single connectivity state per subject (e.g., resting-state or structural connectivity data))';
    warning(msg);
    connMatrix_orig=connMatrix_predSub;
    connMatrix_predSub=zeros(size(connMatrix_orig,1),size(connMatrix_orig,2),1,size(connMatrix_orig,3));
    connMatrix_predSub(:,:,1,:)=connMatrix_orig;
elseif ndims(connMatrix_predSub)<3
    msg='The connectivity matrix should be 4 dimensions (regions X regions X states X subjects)';
    error(msg);
end
if ndims(taskActMatrix_predSub)<3
    msg='The task activity matrix must be 3 dimensions (regions X states/tasks X subjects)';
    error(msg);
end

%extract info from inputs
numTasks=size(taskActMatrix_predSub,2);
numRegions=size(taskActMatrix_predSub,1);
numConnStates=size(connMatrix_predSub,3);
numSubjs=size(connMatrix_predSub,4);

%initiate prediction outputs
taskPredMatrix=zeros(numRegions,numTasks,numSubjs);
taskPredRs=zeros(numTasks,numSubjs);
taskPredPs=zeros(numTasks,numSubjs);
taskActualMatrix=taskActMatrix_predSub;
regionNumList=1:numRegions;

%loop through subjects, tasks and target regions to generate activity flow predictions
for subjNum=1:numSubjs
    for taskNum=1:numTasks

        %pull out group activation template
        taskActVect=taskActMatrix_group(:,taskNum);

        for regionNum=1:numRegions
            %hold out region that is being predicted
            otherRegions=regionNumList;
            otherRegions(regionNum)=[];

            %get this target region's connectivity pattern
            if numConnStates > 1
                stateFCVect=connMatrix_predSub(:,regionNum,taskNum,subjNum);
            else
                %if using resting-state (or any single state) data
                stateFCVect=connMatrix_predSub(:,regionNum,1,subjNum);
            end

            %Calculate activity flow prediction via dot product
            taskPredMatrix(regionNum,taskNum,subjNum)=sum(taskActVect(otherRegions).*stateFCVect(otherRegions));

        end
        
        %Confine predictions to dysfunctional activations (activation
        %differences between the unhealthy subject and the healthy group)
        if run_dysfunc_reg==1
            %regress out group healthy activations from the predicted and
            %actual activations, take residuals for ensuing prediction accuracy assessment
            stats = regstats(taskPredMatrix(:,taskNum,subjNum), taskActVect, 'linear', {'r'});
            taskPredMatrix(:,taskNum,subjNum)=stats.r; %take residuals
            stats = regstats(taskActualMatrix(:,taskNum,subjNum), taskActVect, 'linear', {'r'});
            taskActualMatrix(:,taskNum,subjNum)=stats.r;
        end
        
        %Calculate predicted to actual similarity for this task (i.e. prediction
        %accuracy for this subject)
        [r,p]=corrcoef(taskPredMatrix(:,taskNum,subjNum),taskActualMatrix(:,taskNum,subjNum));
        taskPredRs(taskNum,subjNum)=r(1,2);
        taskPredPs(taskNum,subjNum)=p(1,2);
    end
end

%Compute subject-level prediction accuracy ('compare then average' approach)
%separately for each task
output.r_bytask=tanh(mean(atanh(taskPredRs),2));
output.p_bytask=ones(numTasks,1);
output.t_bytask=ones(numTasks,1);
for taskNum=1:numTasks
    [~, output.p_bytask(taskNum),~,stats]=ttest(atanh(taskPredRs(taskNum,:)));
    output.t_bytask(taskNum)=stats.tstat;  
end
%and for the average across tasks
output.r_overall=tanh(mean(mean(atanh(taskPredRs),1),2));
[~, output.p_overall, ~, stats]=ttest(mean(atanh(taskPredRs),1));
output.t_overall=stats.tstat;

%assign to output
output.r_bysubj=taskPredRs;
output.p_bysubj=taskPredPs;
output.taskActualMatrix=taskActualMatrix;
output.taskPredMatrix=taskPredMatrix;

%Compute group-level prediction accuracy ('average-then-compare' approach)
%separately for each task
output.r_avgfirst_bytask=zeros(numTasks,1);
output.p_avgfirst_bytask=zeros(numTasks,1);
for taskNum=1:numTasks
    [output.r_avgfirst_bytask(taskNum),output.p_avgfirst_bytask(taskNum)]=corr(mean(taskPredMatrix(:,taskNum,:),3),mean(taskActualMatrix(:,taskNum,:),3));
end
output.r_avgfirst_mean=tanh(mean(atanh(output.r_avgfirst_bytask)));
%and for the average across tasks
avg_Pred=mean(taskPredMatrix,3);
avg_Pred=mean(avg_Pred,2);
avg_Actual=mean(taskActualMatrix,3);
avg_Actual=mean(avg_Actual,2);
[~,output.p_avgfirst_mean]=corr(avg_Pred,avg_Actual); 
    
end

