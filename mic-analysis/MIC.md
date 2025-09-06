好的，这是您提供的文档整理后的 Markdown 格式。

-----

# 时滞最大信息系数（MIC）分析：揭示时间序列中的隐藏关联

## 什么是MIC？

继上一篇文章利用最大互信息系数来筛选特征，真香！MATLAB免费代码

\*\*最大互信息系数（MIC）\*\*是一种用于衡量两个变量之间关联强度的统计量，其核心思想是通过动态网格划分，捕捉变量间的复杂关系（线性、非线性、周期性等），并将互信息进行归一化处理，使结果具有可比性。

最大信息系数（Maximum Information Coefficient, MIC）是一种用于衡量变量间非线性关系的统计方法，特别适用于时间序列分析。其核心特点包括：

  * **取值范围：** 0到1之间
  * **接近0：** 变量间几乎独立
  * **接近1：** 存在强非线性关系
  * **优势：** 相比传统相关系数，能捕捉更复杂的依赖模式

## 时滞分析的关键概念

在时间序列分析中，时滞是指当前时刻与未来时刻之间的时间间隔。时滞的引入使得我们能够分析变量在不同时间延迟下的相互关系。常见的时滞情况包括：

1.  **时滞0：** 表示当前时刻的数据与当前时刻的特征之间的关系。
2.  **时滞1：** 表示当前时刻的数据与前一个时刻的特征之间的关系。
3.  **时滞2：** 表示当前时刻的数据与前两个时刻的特征之间的关系，以此类推。

通过引入不同的时滞，分析不同时间延迟下变量之间的关系，能够揭示出某些特征在不同时间段内对目标变量的影响。

## MATLAB实战案例分析

数据集选择是UCI公开数据集：[Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)

该数据集包含 Capital 共享单车系统中 2011 年至 2012 年间每小时和每天的租赁自行车数量，以及相应的天气和季节信息。是做回归预测很好的数据集。

部分数据截图如下，最后一列是租赁数量，前面几列为特征。

我们本期就用此数据集进行实验，MATLAB完整代码如下：

```matlab
%% ==================== 环境初始化 ====================
% 清空工作空间并重置MATLAB环境
warning off      % 关闭警告提示
close all        % 关闭所有图形窗口
clear all        % 清除工作区所有变量
clc              % 清空命令窗口

%% ==================== 结果目录设置 ====================
resultsDir = 'FeatureAnalysisResults'; % 结果保存目录
% 检查并创建结果目录
if ~exist(resultsDir, 'dir')
    mkdir(resultsDir);
    fprintf('创建结果目录: %s\n', resultsDir);
end

%% ==================== 数据导入与预处理 ====================
% 读取共享单车租赁数据集
[rawData, textHeaders, ~] = xlsread('共享单车租赁数据集.csv');

% 数据提取与预处理
featureData = rawData(:, 3:end);  % 从第3列开始提取特征数据
featureNames = textHeaders(1, 3:end)';  % 获取特征名称并转置为列向量
targetVariable = featureData(:, end);  % 最后一列作为目标变量
predictorVariables = featureData(:, 1:end-1);  % 其余列为预测变量
numFeatures = size(predictorVariables, 2);  % 特征数量

%% ==================== 多时滞MIC分析 ====================
% 参数配置
maxLag = 10;  % 最大时间滞后步长
lagSteps = 0:maxLag;  % 滞后步长序列
numLags = length(lagSteps);  % 滞后步长数量

% 初始化MIC结果矩阵
micMatrix = zeros(numFeatures, numLags);  % 存储各特征在不同滞后下的MIC值

% 多时滞MIC计算
for lagIdx = 1:numLags
    currentLag = lagSteps(lagIdx);
    
    % 构建时滞特征矩阵
    if currentLag == 0
        laggedFeatures = predictorVariables;
    else
        % 使用首值填充法处理时滞数据
        initialValues = repmat(predictorVariables(1, :), currentLag, 1);
        laggedFeatures = [initialValues; predictorVariables(1:end-currentLag, :)];
    end
    
    % 计算当前滞后下的MIC值
    micValues = MIC(laggedFeatures, targetVariable);
    micMatrix(:, lagIdx) = micValues;
    
    fprintf('完成滞后%d步的MIC计算...\n', currentLag);
end

%% ==================== MIC热力图可视化 ====================
figure('Units', 'normalized', 'Position', [0.1, 0.1, 0.7, 0.7], ...
       'Color', [0.98, 0.98, 0.98], 'Name', 'MIC热力图分析');
% 创建热力图
h = heatmap(lagSteps, featureNames(1:end-1), micMatrix, ...
            'Colormap', parula, ...
            'ColorScaling', 'scaled', ...
            'ColorLimits', [0, 1]);
colormap(hot); % 设置颜色映射为 hot

% 图表美化
h.Title = '特征时滞MIC相关性分析';
h.XLabel = '滞后步长 (Lag Steps)';
h.YLabel = '特征名称';
h.FontSize = 10;
h.GridVisible = 'off';

% 保存热力图
savePath = fullfile(resultsDir, 'FeatureLagMIC_Heatmap');
exportgraphics(gcf, [savePath '.png'], 'Resolution', 300);
saveas(gcf, [savePath '.fig']);
fprintf('MIC热力图已保存至: %s\n', savePath);

%% ==================== 最大MIC值分析 ====================
% 提取各特征的最大MIC值及对应滞后步长
[maxMICValues, optimalLagIndices] = max(micMatrix, [], 2);
optimalLags = lagSteps(optimalLagIndices)';

% 按MIC值降序排序
[sortedMIC, sortIdx] = sort(maxMICValues, 'descend');
sortedFeatures = featureNames(sortIdx);
sortedLags = optimalLags(sortIdx);

%% ==================== 最大MIC柱状图 ====================
figure('Units', 'normalized', 'Position', [0.1, 0.1, 0.7, 0.8], ...
       'Color', [0.98, 0.98, 0.98], 'Name', '特征最大MIC分析');

% 创建分组柱状图（按MIC值分组着色）
colorGroups = ceil((1:numFeatures)/5);  % 每5个特征一组颜色
colorMap = lines(max(colorGroups));  % 使用lines颜色映射

hold on;
for i = 1:numFeatures
    barh(i, sortedMIC(i), ...
         'FaceColor', colorMap(colorGroups(i), :), ...
         'EdgeColor', 'none', ...
         'BarWidth', 0.8);
end

% 添加数值和滞后信息
textYOffset = 0.15;  % 文本垂直偏移量
for i = 1:numFeatures
    text(sortedMIC(i) + 0.02, i, ...
         sprintf('MIC: %.2f\nLag: %d', sortedMIC(i), sortedLags(i)), ...
         'FontSize', 9, ...
         'VerticalAlignment', 'middle');
end

% 图表美化
set(gca, 'YTick', 1:numFeatures, ...
         'YTickLabel', sortedFeatures, ...
         'TickLabelInterpreter', 'none', ...
         'FontSize', 10, ...
         'XGrid', 'on', ...
         'GridLineStyle', ':', ...
         'GridAlpha', 0.3);

title('特征最大MIC值及时滞分析', ...
      'FontSize', 12, 'FontWeight', 'bold');
xlabel('最大MIC值', 'FontSize', 11, 'FontWeight', 'bold');
xlim([0, min(1.1, max(sortedMIC)*1.2)]);

% 去除图形四周的白边
set(gca, 'LooseInset', get(gca, 'TightInset'));

% 保存柱状图
savePath = fullfile(resultsDir, 'MaxMIC_BarChart');
exportgraphics(gcf, [savePath '.png'], 'Resolution', 300);
saveas(gcf, [savePath '.fig']);
fprintf('最大MIC柱状图已保存至: %s\n', savePath);

%% ==================== 特征选择与重构 ====================
% 参数配置
topN = 8;  % 选择特征个数
selectedIdx = sortIdx(1:topN);  % 选择索引
selectedFeatures = featureNames(selectedIdx);  % 特征名称
selectedLags = optimalLags(selectedIdx);  % 最优滞后步长

% 数据重构
processedData = zeros(size(predictorVariables, 1), topN+1);  % 预分配内存

for i = 1:topN
    featIdx = selectedIdx(i);
    lagVal = selectedLags(i);
    
    % 时滞数据处理
    if lagVal == 0
        processedData(:, i) = predictorVariables(:, featIdx);
    else
        % 使用首值填充缺失数据
        initVal = predictorVariables(1, featIdx);
        processedData(1:lagVal, i) = initVal;
        processedData(lagVal+1:end, i) = predictorVariables(1:end-lagVal, featIdx);
    end
end

% 添加目标变量
processedData(:, end) = targetVariable;

% 输出选择结果
fprintf('\n======== 最优特征选择结果 ========\n');
fprintf('%-20s %-10s %-10s\n', '特征名称', 'MIC值', '最优滞后');
for i = 1:topN
    fprintf('%-20s %-10.3f %-10d\n', ...
            selectedFeatures{i}, sortedMIC(i), selectedLags(i));
end

% 保存处理后的数据
save(fullfile(resultsDir, 'ProcessedData.mat'), 'processedData', 'selectedFeatures');
fprintf('\n处理后的数据已保存至: %s\n', fullfile(resultsDir, 'ProcessedData.mat'));
```

-----

### 运行结果分析

由上图可以看出，**`registered`** 与租赁数量相关性最强，且在时滞为0的时候，相关性最强，这说明了 `registered` 这个特征与当前时刻下的租赁数量相关性是最强的。再看 **`hum`** 这个特征，这个特征就是在时滞为7的时候，与目标量相关性最强。但是整体来看此热力图，还是当时滞为0的时候，特征与目标量的相关性最强。

这可能与本期的数据有关，当换一种目标变量在时间上有延迟性的数据的时候，可能这种时滞MIC分析就能体现出效果了。