using Clustering, Plots,Distances
using CSV
using LinearAlgebra
using DataFrames
using Random
using Statistics
using StatsBase

#建造confusion matrix
function prediction_data(K_num,test_data,centroids)
    for i in 1 : 10 :18000
        println(i)
        global test_prob = []
        global dist = []
        for j in 1 : 8192
            vector_A = collect(test_data[i,:])
            vector_B = collect(centroids[j,:])
            temp_dist = cal_distance(vector_A,vector_B)
            push!(dist ,temp_dist)
        end
        println("The dist: ",dist)
        #創建一個包含值和索引的元组數組並算出權重
        global value_index_pairs = [(value, index) for (index, value) in enumerate(dist)]
        #按值排序
        global sorted_pairs = sort(value_index_pairs, by = x -> x[1])
        #提取前K個最小值的索引
        global K_index = [pair[2] for pair in sorted_pairs[1:K_num]]
        global K_dist = [pair[1] for pair in sorted_pairs[1:K_num]]
        global alpha_prob = cal_alpha(K_dist)
        println("alpha: $alpha_prob ")
        #for j in 1 : 512
            #test_prob = []
            #算given 機率 temp -> train_total_1
        for i in 1:5
            temp_prob = cal_prob(train_data,i,K_index)
            total_prob = sum(alpha_prob .* temp_prob)
            push!(test_prob, total_prob)
        end
        println("The test_prob: ", test_prob)
        push!(frame_class, findmax(test_prob)[2])
    end
    println("The frame class: $frame_class")
    # 使用 countmap 計算每個數字出現的次數
    count_map = countmap(frame_class)
    # 找到出現次數最多的數字
    max_key = findmax(count_map)[2]
    println("The label that you predict: $max_key")
    return max_key
end

#計算有幾個ABCDE
function cal_classnum(center)
    A_num = count(row -> row[:class] == 1, eachrow(center))
    B_num = count(row -> row[:class] == 2, eachrow(center))
    C_num = count(row -> row[:class] == 3, eachrow(center))
    D_num = count(row -> row[:class] == 4, eachrow(center))
    E_num = count(row -> row[:class] == 5, eachrow(center))
    total = [A_num,B_num,C_num,D_num,E_num]
    _,max_index = findmax(total)
    return max_index
end

#建造confusion matrix
function confusion_m(real_label, pred_label,conf_m)
    for i in 1:size(pred_label,1)
        conf_m[pred_label[i], real_label[i]] = conf_m[pred_label[i], real_label[i]] + 1
    end
    
    return conf_m
end

#計算這個centroid裡面ABCDE的機率 centroid[:,1] temp -> train_total_1
function cal_prob(train_total_1,class_i,centroid_i)
    global prob = []
    for i in centroid_i
        centroid_data =  filter(row -> row[:cluster] == i, train_total_1)
        group_num = count(row -> row[:class] == class_i, eachrow(centroid_data))
        p = group_num / size(centroid_data,1)
        push!(prob,p)
    end
    println("The prob: $prob")

    return prob
end

#計算alpha權重
function cal_alpha(dist)
    global alpha = []
    for i in dist
        push!(alpha,(1/i))
    end
    return alpha
end

#計算兩向量距離
function cal_distance(A,B)
    dist = sqrt(sum((A .- B) .^ 2))
    return dist
end

function cal_PCA(data)
    #計算covariance Matrix (6X6)
    Conv = (1/size(data,1)) * (Matrix(data)' * Matrix(data))
    # 計算covariance 的特徵值和特徵向量
    eigen_conv = eigen(Conv)

    # 取得特徵值與特真與特徵向量
    #eigen_values = eigen_conv.values
    eigen_vector = eigen_conv.vectors

    #reduce_eigen_values = eigen_values[1:20]
    #投影矩陣
    global Projection = eigen_vector[1,:]
    for i in 2:10
        global Projection = hcat(Projection,eigen_vector[i,:])
    end

    #計算出最後的轉換函數Y
    Y = Matrix(data) * Projection
    return Y
end

function cal_standard(train_data)
    for i in 1:26
        column_data = train_data[!, i]
        std_dev = std(column_data)
        mean_data = mean(column_data)
        train_data[!, i] .= (column_data .- mean_data) / std_dev
    end
    return train_data
end


train_data = CSV.read("data_with_clusters.csv", DataFrame)
#assignments = CSV.read("assignments_1.csv", DataFrame)
#assignments = assignments .+ 1
#assignments_new = vec(Matrix(assignments))
centroids = CSV.read("clustered_data.csv",DataFrame)
centroids = centroids[:,1:10]

# 使用 insertcols! 将聚类数据向量添加为新的列
#insertcols!(train_data, :cluster => assignments_new)


# global real_label_A = [1 for i in 1:65]
# global real_label_B = [2 for i in 1:101]
# global real_label_C = [3 for i in 1:395]
# global real_label_D = [4 for i in 1:51]
# global real_label_E = [5 for i in 1:395]
global confusion_matrix =  [0 for i in 1:5, j in 1:5]
global real_label_A = [1 for i in 1:1800]
global real_label_B = [2 for i in 1:1800]
global real_label_C = [3 for i in 1:1800]
global real_label_D = [4 for i in 1:1800]
global real_label_E = [5 for i in 1:1800]
# global cluster_class = []
# #算centroid的class
# for i in 1: 512
#     group_data = filter(row -> row[:cluster] == i, train_data)
#     push!(cluster_class,cal_classnum(group_data))
# end
# println(cluster_class)
#KNN
global K_num = 3 #決定鄰居數
global dist_min = []
global frame_class = []


global predict_label_A = []
global predict_label_B = []
global predict_label_C = []
global predict_label_D = []
global predict_label_E = []
#輸入A test資料
for i in 22: 24#385
    # 生成檔案名稱，類似 "0001.csv"
    #global predict_label_A = []
    filename = "Dataset3/A_MFCC/test/" * lpad(i, 4, '0') * ".csv"  
    # 讀取 CSV 檔案
    test_data = CSV.read(filename, DataFrame) 
    test_data = cal_standard(test_data)
    test_data = cal_PCA(test_data)
    test_data = DataFrame(test_data, :auto)
    temp_label = prediction_data(K_num,test_data,centroids)
    push!(predict_label_A, temp_label)
    println("Confusion Matrix:")
    global conf_matrix = confusion_m(real_label_A,predict_label_A,confusion_matrix)
    println(conf_matrix)
end
println(predict_label_A)


# 輸入B test資料
for i in 1:2 #385
    # 生成檔案名稱，類似 "0001.csv"
    #global predict_label_B = []
    filename = "Dataset3/B_MFCC/test/" * lpad(i, 4, '0') * ".csv"  
    # 讀取 CSV 檔案
    test_data = CSV.read(filename, DataFrame) 
    test_data = cal_standard(test_data)
    test_data = cal_PCA(test_data)
    test_data = DataFrame(test_data, :auto)
    temp_label = prediction_data(K_num,test_data,centroids)
    push!(predict_label_B, temp_label)
    println("Confusion Matrix:")
    global conf_matrix = confusion_m(real_label_B,predict_label_B,confusion_matrix)
    println(conf_matrix)
end

# 輸入C test資料
for i in 5:6 #385
    # 生成檔案名稱，類似 "0001.csv"
    #global predict_label_C = []
    filename = "Dataset3/C_MFCC/test/" * lpad(i, 4, '0') * ".csv"  
    # 讀取 CSV 檔案
    test_data = CSV.read(filename, DataFrame) 
    test_data = cal_standard(test_data)
    test_data = cal_PCA(test_data)
    test_data = DataFrame(test_data, :auto)
    temp_label = prediction_data(K_num,test_data,centroids)
    push!(predict_label_C, temp_label)
    println("Confusion Matrix:")
    global conf_matrix = confusion_m(real_label_C,predict_label_C,confusion_matrix)
    println(conf_matrix)
end

# 輸入D test資料
for i in 1:10 #385
    # 生成檔案名稱，類似 "0001.csv"
    #global predict_label_D = []
    filename = "Dataset3/D_MFCC/test/" * lpad(i, 4, '0') * ".csv"  
    # 讀取 CSV 檔案
    test_data = CSV.read(filename, DataFrame) 
    test_data = cal_standard(test_data)
    test_data = cal_PCA(test_data)
    test_data = DataFrame(test_data, :auto)
    temp_label = prediction_data(K_num,test_data,centroids)
    push!(predict_label_D, temp_label)
    println("Confusion Matrix:")
    global conf_matrix = confusion_m(real_label_D,predict_label_D,confusion_matrix)
end

# 輸入E test資料
for i in 1:10 #385
    # 生成檔案名稱，類似 "0001.csv"
    #global predict_label_E = []
    filename = "Dataset3/E_MFCC/test/" * lpad(i, 4, '0') * ".csv"  
    # 讀取 CSV 檔案
    test_data = CSV.read(filename, DataFrame) 
    test_data = cal_standard(test_data)
    test_data = cal_PCA(test_data)
    test_data = DataFrame(test_data, :auto)
    temp_label = prediction_data(K_num,test_data,centroids)
    push!(predict_label_E, temp_label)
    println("Confusion Matrix:")
    global conf_matrix = confusion_m(real_label_E,predict_label_E,confusion_matrix)
end
# real_label_AB = vcat(real_label_A, real_label_B)
# predict_label_AB = vcat(predict_label_A,predict_label_B)
# real_label_CD = vcat(real_label_C,real_label_D)
# predict_label_CD = vcat(predict_label_C,predict_label_D)
# real_label_ABCD = vcat(real_label_AB,real_label_CD)
# predict_label_ABCD = vcat(predict_label_AB,predict_label_CD)
# predict_label = vcat(predict_label_ABCD,predict_label_E)
# real_label = vcat(real_label_ABCD,real_label_E)

#confusion_matrix = confusion_m(real_label_A,predict_label_A,confusion_matrix)
println("Final Confusion Matrix:")
println(confusion_matrix)
println("------------")
acc = confusion_matrix[1,1] + confusion_matrix[2,2] +confusion_matrix[3,3] + confusion_matrix[4,4] + confusion_matrix[5,5]
println("Accuracy: ",acc/(sum(confusion_matrix)))