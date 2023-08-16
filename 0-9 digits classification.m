clc; clear;
 
%% ======================= Parameters ===========================
N = 4000; % number of images of each digit to train and test
part = 5; % part*N is the number of images for traning

%%  ==================== Load MNIST dataset ======================
load('mnist.mat');

imagesPerDigit = zeros(28,28,N,10);
for digit=0:1:9
    currImagesPerDigit = training.images(:,:,training.labels == digit);
    imagesPerDigit(:,:,:,digit+1) = currImagesPerDigit(:,:,1:N);
end

%% ======================= Create A, b ============================
% 10*N rows, N rows for each digit from 0 to 9
A_all = zeros(10*N,28^2);
% create bk for each digit k-1 from 0 to 9 such that bk has +1 in rows
% matching images of i and -1 in the other rows
% each bk is a vector with 10*N rows and one col
b_all_matrix = zeros(10*N,10);
% put the images in A
j = 1;
for i=1:10:(10*N-9)
    for digit=0:1:9
        A_all(digit+i,:) = reshape(imagesPerDigit(:,:,j,digit+1),1,28*28);
        for k=1:1:10
            if digit+1 == k
                b_all_matrix(digit+i,k)   = 1;
            else
                b_all_matrix(digit+i,k)   = -1;
            end
        end
    end
    j = j+1;
end

% add to A the ones col (to match to c)
A_all = [A_all, ones(10*N,1)];

%% ========================= Solve LS ==============================
% we train only on the first 5*N rows of A (so we get N/2 images of each 
% digit to train on)
A_train = A_all(1:part*N,:); 
b_train_matrix = b_all_matrix(1:part*N,:); 

% calculate xk for each digit k from 0 to 9
x_matrix = zeros(785,10);
% lambda is a scalar we choose for the simple regularization
lambda = 1;

for k=1:1:10
    % x_matrix(:,k) = inv(A_train'*A_train+lambda*eye(785))*A_train'*b_train_matrix(:,k); 
    x_matrix(:,k) = (A_train'*A_train+lambda*eye(785))\(A_train'*b_train_matrix(:,k)); 
end

% the last 5*N rows of A will be used for testing
A_test = A_all(part*N+1:10*N,:); 
b_test_matrix = b_all_matrix(part*N+1:10*N,:); 

%% ===================== Check Performance ===========================
% test each xk we got on the images we trained on
for k=1:1:10
    disp(['digit: ',num2str(k-1)]);
    predC = sign(A_train*x_matrix(:,k)); 
    trueC = b_train_matrix(:,k); 
    disp('Train Error:');
    % get the precentage of the right predictions
    acc=mean(predC == trueC)*100;
    disp(['Accuracy=',num2str(acc),'% (',num2str((1-acc/100)*(part*N)),' wrong examples)']); 
end

% check accuracy in the test images (which didn't help in calculating x)
disp('------------------------')
for k=1:1:10
    disp(['digit: ',num2str(k-1)]);
    predC = sign(A_test*x_matrix(:,k)); 
    trueC = b_test_matrix(:,k); 
    disp('Test Error:'); 
    acc=mean(predC == trueC)*100;
    disp(['Accuracy=',num2str(acc),'% (',num2str((1-acc/100)*((10-part)*N)),' wrong examples)']); 
end
%% ================= Show the Problematric Images ====================
% find function gives us a vector containing all indices of images in 
% A_test we got wrong

for digit=0:1:9
    predC = sign(A_test*x_matrix(:,digit+1));
    trueC = b_test_matrix(:,digit+1); 
    error = find(predC~=trueC); 
    % show 1 image 
    figure(2);
    % print image row as and 28*28 image
    k = 1;
    if digit == 8
        k = 2;
    end
    imagesc(reshape(A_test(error(k),1:28^2),[28,28]));
    colormap(gray(256))
    axis image; axis off;
    % print image number and (image_row)*x to see what value we got 
    title(['digit to identify: ',num2str(digit),'. Problematic digit number ',num2str(k),' :',num2str(A_test(error(k),:)*x_matrix(:,digit+1))]); 
    pause;  
end


