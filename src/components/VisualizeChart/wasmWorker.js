/* eslint-disable no-restricted-globals */
/* eslint-disable no-unused-vars */
/* eslint-disable new-cap */

import get from 'lodash/get';
import init, { initThreadPool, bhtSNEf64 } from 'wasm-bhtsne';
import { threads } from 'wasm-feature-detect';

const errorMessage = {
    data: [],
    error: 'No data found',
};

const MESSAGE_INTERVAL = 200;

self.onmessage = e => {
    let lastTime = new Date().getTime();
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
        // points.forEach(point => data.push(...point.vector));
        points.forEach(point => data.push(point.vector));
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
        // points.forEach(point => data.push(...point.vector[vecName]));
        points.forEach(point => data.push(point.vector[vecName]));
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
            if (await threads()) {
                console.log("Browser supports threads");
                await initThreadPool(navigator.hardwareConcurrency);
            }
            else {
                console.log("Browser does not support threads");
            }

            // const vectors = [];
            // for (let i = 0; i < data.length; i += cols) {
            //     const chunk = data.slice(i, i + cols);
            //     vectors.push(chunk);
            // }

            // set hyperparameters
            const opt = {
                learning_rate: 150.0,
                perplexity: 30.0,
                theta: 0.6,
            };
            try {
                const tsneEncoder = new bhtSNEf64(data, opt);
                let result;
                for (let i = 0; i < 1000; i++) {
                    result = tsneEncoder.step(1);

                    if (Date.now() - lastTime < MESSAGE_INTERVAL) {
                        continue;
                    }

                    lastTime = Date.now();
                    self.postMessage({
                        result: getDataset(e.data, result, 2),
                        error: null,
                    });
                }
                // const tsne = tSNEf32.new(data, cols);
                // tsne.perplexity(1.0);
                // tsne.epochs(500);

                // const result = tsne.barnes_hut(0.5);
                // console.log(data);
                // console.log(result);
            } catch (error) {
                self.postMessage({
                    data: [],
                    error: error,
                });
            }
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
            // dataset[data.labelByArrayUnique.indexOf(label)].data.push({
            //     x: reducedPoint[idx * cols + 0],
            //     y: reducedPoint[idx * cols + 1],
            //     point: point,
            // })
            dataset[data.labelByArrayUnique.indexOf(label)].data.push({
                x: reducedPoint[idx][0],
                y: reducedPoint[idx][1],
                point,
            })
        })
    }
    else {
        dataset.push({
            label: 'data',
            data: [],
        })
        points?.forEach((point, idx) => {
            // dataset[0].data.push({
            //     x: reducedPoint[idx * cols + 0],
            //     y: reducedPoint[idx * cols + 1],
            //     point,
            // })
            dataset[0].data.push({
                x: reducedPoint[idx][0],
                y: reducedPoint[idx][1],
                point,
            })
        })
    }

    return dataset;
}