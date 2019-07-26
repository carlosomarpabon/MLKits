require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('../load-csv.js');
const LogisticRegression = require('./logistic-regression');
const plotly = require('plotly')("Warpriest86", "NUIBixNsIrbYGiulQLIJ");

const { features, labels, testFeatures, testLabels } = loadCSV(
    '../data/cars.csv',
    {
        dataColumns: ['horsepower', 'displacement', 'weight'],
        labelColumns: ['passedemissions'],
        shuffle: true,
        splitTest: 50,
        converters: {
            passedemissions: value => {
                return value === 'TRUE' ? 1 : 0;
            }
        }
    }
);

const regression = new LogisticRegression(features, labels, {
    learningRate: 0.5,
    iterations: 100,
    batchSize: 50,
    decisionBoundary: 0.5
});

regression.train();

console.log(regression.test(testFeatures, testLabels));

var data = [{
    y: regression.costHistory.reverse(),
    type: "scatter"
}];
var layout = { fileopt: "overwrite", filename: "CostHistory" };
plotly.plot(data, layout, function (err, msg) {
    if (err) return console.log(err);
    console.log(msg);
});