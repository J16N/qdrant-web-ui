/* eslint-disable no-restricted-globals */
/* eslint-disable no-unused-vars */
/* eslint-disable new-cap */
/* eslint-disable prefer-const */

import get from 'lodash/get';
import init, { initThreadPool, bhtSNEf64 } from 'wasm-bhtsne';
import { threads } from 'wasm-feature-detect';

const errorMessage = {
    data: [],
    error: 'No data found',
};

const MESSAGE_INTERVAL = 200;
let port;
let RENDERING = false;
const sleep = ms => new Promise(r => setTimeout(r, ms));

self.onmessage = e => {
    if (e.data.command === "CONN") {
        port = e.ports[0];
        port.onmessage = event => {
            RENDERING = event.data;
        };
        return;
    }
    // let lastTime = new Date().getTime();
    const data = [];
    const points = e.data.details.result?.points;
    let vector;
    let vecName;
    let cols = 0;
    let outputDim = 2;

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
            const { memory } = await init();
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
                for (let i = 0; i < 1000; i++) {
                    const resultPtr = tsneEncoder.step(1);

                    // Give chance to other incoming messages
                    await sleep(0);

                    if (RENDERING) continue;

                    const result = new Float64Array(memory.buffer, resultPtr, points.length * outputDim);
                    self.postMessage({
                        result: getDataset(e.data.details, result, outputDim),
                        error: null,
                    });

                    RENDERING = true;
                }
            }
            catch (error) {
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
            dataset[data.labelByArrayUnique.indexOf(label)].data.push({
                x: reducedPoint[idx * cols + 0],
                y: reducedPoint[idx * cols + 1],
                point: point,
            })
            // dataset[data.labelByArrayUnique.indexOf(label)].data.push({
            //     x: reducedPoint[idx][0],
            //     y: reducedPoint[idx][1],
            //     point,
            // })
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
            // dataset[0].data.push({
            //     x: reducedPoint[idx][0],
            //     y: reducedPoint[idx][1],
            //     point,
            // })
        })
    }

    return dataset;
}