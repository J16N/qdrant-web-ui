/* eslint-disable no-restricted-globals */
/* eslint-disable no-unused-vars */
/* eslint-disable new-cap */
/* eslint-disable prefer-const */

import init, { initThreadPool, Tsne } from 'wasm_bhtsne';
import { threads } from 'wasm-feature-detect';

const errorMessage = {
    data: [],
    error: 'No data found',
};

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

    console.log(`Received data in: ${(Date.now() - e.data.time) / 1000}s`);
    const data = [];
    const points = e.data.details.result?.points;
    let vector;
    let vecName;
    let cols = 0;
    let outputDim = 2;
    const sharedArray = new Float64Array(e.data.sharedArray);

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
        // points.forEach(point => data.push(point.vector));
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
        // points.forEach(point => data.push(point.vector[vecName]));
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

            // set hyperparameters
            const opt = {
                learning_rate: 150.0,
                perplexity: 30.0,
                theta: 1.5,
            };

            try {
                console.time('Rust rewrite - t-SNE Total Time');
                console.time('Rust rewrite - t-SNE 1st step');
                const tsne = new Tsne(data, cols);
                console.timeEnd('Rust rewrite - t-SNE 1st step');

                console.time('Rust rewrite - t-SNE 2nd step');
                for (let i = 0; i < 500; i++) {
                    tsne.step();

                    // Give chance to other coroutines to run
                    await sleep(0);

                    if (RENDERING) continue;
                    sendVisual(self, sharedArray, tsne, memory, points, outputDim);
                    RENDERING = true;
                }
                console.timeEnd('Rust rewrite - t-SNE 2nd step');
                sendVisual(self, sharedArray, tsne, memory, points, outputDim);
                console.timeEnd('Rust rewrite - t-SNE Total Time');
            }
            catch (error) {
                console.error(error);
                self.postMessage({
                    data: [],
                    error: error,
                });
            }
        })();
    }
};

self.onerror = e => {
    console.error(e);
};

function sendVisual(worker, sharedArray, tsne, memory, points, outputDim) {
    let resultPtr = tsne.embedding();
    const result = new Float64Array(memory.buffer, resultPtr, points.length * outputDim);
    result.forEach((val, idx) => {
        sharedArray[idx] = val;
    });

    worker.postMessage({
        error: null,
    });
}