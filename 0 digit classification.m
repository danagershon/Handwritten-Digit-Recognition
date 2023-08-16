clc; clear;
 
%% ======================= Parameters ===========================
N = 4000; % number of images of each digit to train and test

%%  ==================== Load MNIST dataset ======================
load('mnist.mat');

% for each of the 10 digits get the first N images from training images
imagesPerDigit = zeros(28,28,N,10);
for digit=0:1:9
    % gets us about 6000 images
    currImagesPerDigit = training.images(:,:,training.labels == digit);
    % keep only N images
    imagesPerDigit(:,:,:,digit+1) = currImagesPerDigit(:,:,1:N);
end

%% ======================= Create A, b ============================
% N rows for each digit from 0 to 9
A_all = zeros(10*N,28^2);
% N rows match to zero images so value will be 1
% other rows match to none zero images so value will be changed to -1
b_all = zeros(10*N,1);
% put the images in A: 0 image, 1 image,...,9 image,0 image and so on
j = 1;
for i=1:10:(10*N-9)
    for digit=0:1:9
        A_all(digit+i,:) = reshape(imagesPerDigit(:,:,j,digit+1),1,28*28);
        if digit == 0
            b_all(digit+i) = 1;
        else
            b_all(digit+i) = -1;
        end
    end
    j = j+1;
end

% add to A the ones col (to match to c)
A_all = [A_all, ones(10*N,1)];

%% ========================= Solve LS ==============================
% we train only on the first 5*N rows of A (so we get N/2 images of each 
% digit to train on)
A_train = A_all(1:5*N,:); 
b_train = b_all(1:5*N); 

% x is [w c]^T, the solution to the LS problem with A and b
% calculate x with the known formula:
% x = pseudoinverse(A)*b
x=pinv(A_train)*b_train; 

% the last 5*N rows of A will be used for testing
A_test = A_all(5*N+1:10*N,:); 
b_test = b_all(5*N+1:10*N); 

%% ===================== Check Performance ===========================
% we test the x we got on the images we trained on
predC = sign(A_train*x); 
trueC = b_train; 
disp('Train Error:');
% get the precentage of the right predictions
acc=mean(predC == trueC)*100;
disp(['Accuracy=',num2str(acc),'% (',num2str((1-acc/100)*(5*N)),' wrong examples)']); 

% check accuracy in the test images (which didn't help in calculating x)
predC = sign(A_test*x); 
trueC = b_test; 
disp('Test Error:'); 
acc=mean(predC == trueC)*100;
disp(['Accuracy=',num2str(acc),'% (',num2str((1-acc/100)*(5*N)),' wrong examples)']); 

%% ================= Show the Problematric Images ====================
% find function gives us a vector containing all indices of images in 
% A_test we got wrong
error = find(predC~=trueC); 
for k=1:1:5
    figure(2);
    % print image row as and 28*28 image
    imagesc(reshape(A_test(error(k),1:28^2),[28,28]));
    colormap(gray(256))
    axis image; axis off; 
    % print image number and (image_row)*x to see what value we got 
    title(['problematic digit number ',num2str(k),' :',num2str(A_test(error(k),:)*x)]); 
    pause;  
end
