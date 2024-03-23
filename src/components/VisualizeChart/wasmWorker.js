/* eslint-disable no-restricted-globals */
/* eslint-disable no-unused-vars */
import get from 'lodash/get';
import init, { tSNEf32, initThreadPool } from 'wasm-bhtsne';

const errorMessage = {
    data: [],
    error: 'No data found',
};

self.onmessage = e => {
    const data = [];
    const points = e.data?.result?.points;
    let vector;
    let vecName;
    let cols = 0;

    if (points?.length === 0) {
        self.postMessage(errorMessage);
        return;
    }
    else if (points?.length === 1) {
        errorMessage.error = 'cannot perform tsne on single point';
        self.postMessage(errorMessage);
    }
    else if (typeof (vector = points[0].vector).length === 'number') {
        cols = vector.length;
        points.forEach(point => data.push(...point.vector));
    }
    else if (typeof vector === 'object') {
        if (!(vecName = e.data.vector_name)) {
            errorMessage.error = 'No vector name found, select a valid vector_name';
            self.postMessage(errorMessage);
            return;
        }
        else if (vector[vecName] === undefined) {
            errorMessage.error = `No vector found with name ${vecName}`;
            self.postMessage(errorMessage);
            return;
        }
        else if (!vector[vecName]) {
            errorMessage.error = 'Unexpected Error Occurred';
            self.postMessage(errorMessage);
            return;
        }

        if (!Array.isArray(vector[vecName])) {
            errorMessage.error = 'Vector visualization is not supported for sparse vector';
            self.postMessage(errorMessage);
            return;
        }

        cols = vector[vecName].length;
        points.forEach(point => data.push(...point.vector[vecName]));
    }
    else {
        errorMessage.error = 'Unexpected Error Occurred';
        self.postMessage(errorMessage);
        return;
    }

    if (data.length) {
        // Perform t-SNE
        (async () => {
            await init();
            await initThreadPool(navigator.hardwareConcurrency);
            const tsne = tSNEf32.new(data, cols);
            tsne.perplexity(1.0);
            tsne.epochs(2000);
            const result = tsne.barnes_hut(0.5);
            // console.log(data);
            // console.log(result);
            self.postMessage({
                result: getDataset(e.data, result, 2),
                error: null,
            });
        })();
    }
}

self.onerror = e => {
    console.log(e);
}

function getDataset(data, reducedPoint, cols) {
    const dataset = [];
    const labelBy = data.color_by?.payload;
    const points = data.result?.points;

    if (labelBy) {
        data.labelByArrayUnique.forEach(label => {
            dataset.push({
                label: label,
                data: [],
            })
        })
        points?.forEach((point, idx) => {
            const label = get(point.payload, labelBy);
            dataset[data.labelByArrayUnique.indexOf(label)].data.push({
                x: reducedPoint[idx * cols + 0],
                y: reducedPoint[idx * cols + 1],
                point: point,
            })
        })
    }
    else {
        dataset.push({
            label: 'data',
            data: [],
        })
        points?.forEach((point, idx) => {
            dataset[0].data.push({
                x: reducedPoint[idx * cols + 0],
                y: reducedPoint[idx * cols + 1],
                point,
            })
        })
    }

    return dataset;
}