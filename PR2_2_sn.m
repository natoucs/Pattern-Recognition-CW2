
wine=importfile('wine.data.csv', 1, 178);


data=table2array(wine);
test=data(data(:,1)==2,2:end);
train=data(data(:,1)==1,2:end);

test(:,2:end)=normc(test(:,2:end));
train(:,2:end)=normc(train(:,2:end));

test(:,2:end)=zscore(test(:,2:end));
train(:,2:end)=zscore(train(:,2:end));



k=3;



error_euclidean=kmean_euclidean(train,test,error)

error_cityblock=kmean_cityblock(train,test,error)

error_cosine=kmean_cosine(train,test,error)

error_correlation=kmean_correlation(train,test,error)

     

        