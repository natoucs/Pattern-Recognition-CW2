
wine=importfile('wine.data.csv', 1, 178);


data=table2array(wine);
test=data(data(:,1)==2,2:end);
train=data(data(:,1)==1,2:end);

test(:,2:end)=zscore(test(:,2:end));
train(:,2:end)=zscore(train(:,2:end));

k=1003;



error_euclidean=zeros(1,k-2);
error_cityblock=zeros(1,k-2);
error_cosine=zeros(1,k-2);
error_correlation=zeros(1,k-2);

for j=3:k

error_euclidean(1,j-2)=kmean_euclidean(train,test,error_euclidean,3);

error_cityblock(1,j-2)=kmean_cityblock(train,test, error_cityblock,3);

error_cosine(1,j-2)=kmean_cosine(train,test,error_cosine,3);

error_correlation(1,j-2)=kmean_correlation(train,test,error_correlation,3);  

end    

average_euclidean = mean(error_euclidean)
max_euclidean=max(error_euclidean)
min_euclidean=min(error_euclidean)

average_cityblock = mean(error_cityblock)
max_cityblock=max(error_cityblock)
min_cityblock=min(error_cityblock)

average_cosine = mean(error_cosine)
max_cosine=max(error_cosine)
min_cosine=min(error_cosine)

average_correlation = mean(error_correlation)
max_correlation=max(error_correlation)
min_correlation=min(error_correlation)



