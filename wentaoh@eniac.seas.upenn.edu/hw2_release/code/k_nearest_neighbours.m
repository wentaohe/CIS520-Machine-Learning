function labels = k_nearest_neighbours(Xtrain,Ytrain,Xtest,K,distfunc)

    % Function to implement the K nearest neighbours algorithm on the given
    % dataset.
    % Usage: labels = k_nearest_neighbours(Xtrain,Ytrain,Xtest,K)
    
    % Xtrain : N x P Matrix of training data, where N is the number of
    %   training examples, and P is the dimensionality (number of features)
    % Ytrain : N x 1 Vector of training labels (0/1)
    % Xtest : M x P Matrix of testing data, where M is the number of
    %   testing examples.
    % K : number of nearest neighbours used to make predictions on the test
    %     dataset. Remember to take care of corner cases.
    % distfunc: distance function to be used - l1, l2, linf.
    % labels : return an M x 1 vector of predicted labels for testing data.
    
    % YOUR CODE GOES HERE.
    if nargin < 5
        distFunc = 'l2';
    end
    
    numoftestdata = size(Xtest, 1);
    numoftrainingdata = size(Xtrain, 1);
    
    if size(Xtest, 2) ~= size(Xtrain, 2)
        error('Test points and train points do not have the same dimensionality.');
    end
    
    dimensions = size(Xtest, 2);
   
    %Calculate the difference between test point and train point.
    %We want to format both Xtrain and Xtest into matrices that have the
    %same dimension so that the difference between each point can be
    %calculated using a simple matrix subtraction.
    
    %Formatting both Xtrain and Xtest into M x P x N matrices.
    
    %Formatting training data into 1 x P x N first
    training_points = reshape(Xtrain', [1 dimensions numoftrainingdata]);
    
    %Formatting training data into M x P x N (1 x P x N into M x 1 x 1)
    training_points = repmat(training_points, [numoftestdata 1 1]);
    
    %Formatting test data into M x P x N (M x P into 1 x 1 x N)
    test_points = repmat(Xtest, [1 1 numoftrainingdata]);
    
    distance = test_points - training_points;
    
    if distFunc == 'l1'
        distance = sum(abs(distance), 2);
    elseif distFunc == 'l2'
        distance = sqrt(sum(distance.^2, 2));
    elseif distFunc == 'linf'
        distance = max(abs(distance), [], 2);
    else 
        error('Distance Function Error');
    end
    
    %The above calculation changes 'distance' from M x P x N to M x 1 x N,
    %and we have to reformat it to a M x N matrix and then sort the value
    %of each row to find the nearest neighbor.
    
    distance = squeeze(distance);
    % If there is only one test point, we have to convert it back to row
    % vector.
    if numoftestdata == 1
        distance = distance';
    end
    
    nearest_neighbor = sort(distance, 2)(:,1:K);
    
    labels = Ytrain(nearest_neighbor);
    
    if size(nearest_neighbor, 1) == 1
        labels = labels'
    end
    
    if K > 1
        labels = mean(labels, 2);
    end
    
    
end