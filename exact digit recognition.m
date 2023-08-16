%% ====================== Prepare New Test Set ======================
load('mnist.mat');

num_images = test.count;
new_test_images = shiftdim(test.images, 2);
A_new_test = reshape(new_test_images,num_images,28*28);
A_new_test = [A_new_test, ones(num_images,1)];
true_labels = test.labels;

%% ============================ Predict ==============================
UNCLASSIFIED = -1;
pred = UNCLASSIFIED * ones(num_images, 1);

% TODO: compute your predictions

% res(i,j) is the classification of image i as digit j (should be posotive 
% if image i is the number j and negative otherwise)
res = A_new_test*x_matrix;

for i=1:1:num_images
    % for image i find all digits for which the classificat is positive
    matches = find(res(i,:) > 0);
    % if there is only one positive classification, determine that image i
    % is that digit. If there is none or more than one positive 
    % classification then we can't deterime what is the digit 
    if size(matches,2) == 1
        pred(i) = matches(1)-1;
    end
end

error = find(pred~=true_labels); 
for k=1:1:5  
    figure(2);
    % print image row as and 28*28 image
    imagesc(reshape(A_new_test(error(k),1:28^2),[28,28]));
    colormap(gray(256))
    axis image; axis off;
    title(['digit true label: ',num2str(true_labels(error(k))),'. Digit pred: ',num2str(pred(error(k))),'. Problematic digit number ',num2str(k)]); 
    pause;  
end

%% =========================== Evaluate ==============================
acc = mean(pred == true_labels)*100;
disp(['Accuracy=',num2str(acc),'% (',num2str((1-acc/100)*num_images),' wrong examples)']);
