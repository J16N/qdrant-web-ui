/* eslint-disable no-unused-vars */

import { Button } from '@mui/material';
import Chart from 'chart.js/auto';
import chroma from 'chroma-js';
import get from 'lodash/get';
import { useSnackbar } from 'notistack';
import PropTypes from 'prop-types';
import React, { useEffect, useState } from 'react';
import ViewPointModal from './ViewPointModal';
import { imageTooltip } from './ImageTooltip';
import { bigIntJSON } from '../../common/bigIntJSON';

const SCORE_GRADIENT_COLORS = ['#EB5353', '#F9D923', '#36AE7C'];

function prepareDataset(data) {
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
        points?.forEach((point) => {
            const label = get(point.payload, labelBy);
            dataset[data.labelByArrayUnique.indexOf(label)].data.push({
                x: 0,
                y: 0,
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
        points?.forEach((point) => {
            dataset[0].data.push({
                x: 0,
                y: 0,
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

function mutateDataset(dataset, reducedPoint, cols) {
    for (let i = 0; i < reducedPoint.length / cols; ++i) {
        dataset[0].data[i].x = reducedPoint[i * cols];
        dataset[0].data[i].y = reducedPoint[i * cols + 1];
    }
}

// const WASM_SUPPORTED = (() => {
//   try {
//     if (typeof WebAssembly === "object" &&
//       typeof WebAssembly.instantiate === "function") {
//       const module = new WebAssembly.Module(
//         Uint8Array.of(
//           0x0, 0x61, 0x73, 0x6d,
//           0x01, 0x00, 0x00, 0x00
//         )
//       );
//       if (module instanceof WebAssembly.Module)
//         return new WebAssembly.Instance(module) instanceof WebAssembly.Instance;
//     }
//   }
//   catch (e) {
//     console.log(e);
//   }
//   return false;
// })();

// if (wasmSupported()) {
//   WASM_SUPPORTED = true;
//   console.log('WebAssembly supported');
//   await init();
//   await initThreadPool(navigator.hardwareConcurrency);
//   greet(new Array(25).fill(0), 2);
// }

const VisualizeChart = ({ scrollResult }) => {
    const { enqueueSnackbar, closeSnackbar } = useSnackbar();
    const [openViewPoints, setOpenViewPoints] = useState(false);
    const [viewPoints, setViewPoint] = useState([]);
    const action = (snackbarId) => (
        <Button
            variant="outlined"
            color="inherit"
            onClick={() => {
                closeSnackbar(snackbarId);
            }}
        >
            Dismiss
        </Button>
    );

    useEffect(() => {
        if (!scrollResult.data && !scrollResult.error) {
            return;
        }

        if (scrollResult.error) {
            enqueueSnackbar(`Visualization Unsuccessful\nError: ${bigIntJSON.stringify(scrollResult.error)}`, {
                variant: 'error',
                style: { whiteSpace: 'pre-line' },
                action,
            });
            return;
        }
        else if (!scrollResult.data?.result?.points.length) {
            enqueueSnackbar(`Visualization Unsuccessful\nError: No data returned`, {
                variant: 'error',
                style: { whiteSpace: 'pre-line' },
                action,
            });
            return;
        }

        const dataset = [];
        const colorBy = scrollResult.data.color_by;
        const channel = new MessageChannel();

        let labelby = null;
        if (colorBy?.payload) {
            labelby = colorBy.payload;
            // Color and label by payload field
            if (get(scrollResult.data.result?.points[0]?.payload, labelby) === undefined) {
                enqueueSnackbar(`Visualization Unsuccessful\nError: Color by field ${labelby} does not exist`, {
                    variant: 'error',
                    style: { whiteSpace: 'pre-line' },
                    action,
                });
                return;
            }
            scrollResult.data.labelByArrayUnique = [
                ...new Set(scrollResult.data.result?.points?.map((point) => get(point.payload, labelby))),
            ];
            scrollResult.data.labelByArrayUnique.forEach((label) => {
                dataset.push({
                    label: label,
                    data: [],
                });
            });
        } else if (colorBy?.discover_score) {
            // Color by discover score
            const scores = scrollResult.data.result?.points.map((point) => point.score);
            const minScore = Math.min(...scores);
            const maxScore = Math.max(...scores);

            const colorScale = chroma.scale(SCORE_GRADIENT_COLORS);
            const scoreColors = scores.map((score) => {
                const normalizedScore = (score - minScore) / (maxScore - minScore);
                return colorScale(normalizedScore).hex();
            });

            const pointRadii = scrollResult.data.result?.points.map((point) => {
                if (point.from_query) {
                    return 4;
                } else {
                    return 3;
                }
            });

            dataset.push({
                label: 'Discover scores',
                pointBackgroundColor: scoreColors,
                pointBorderColor: scoreColors,
                pointRadius: pointRadii,
                data: [],
            });
        } else {
            // No special coloring
            dataset.push({
                label: 'Data',
                data: [],
            });
        }
        const ctx = document.getElementById('myChart');
        const myChart = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: dataset,
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        grid: {
                            display: false,
                        },
                        display: false,
                    },
                    y: {
                        display: false,
                    },
                },
                plugins: {
                    tooltip: {
                        // only use custom tooltip if color by is not discover score
                        enabled: !colorBy?.discover_score,
                        external: (colorBy?.discover_score && imageTooltip) || undefined,
                        usePointStyle: true,
                        callbacks: {
                            label: (context) => {
                                const payload = bigIntJSON
                                    .stringify(context.dataset.data[context.dataIndex].point.payload, null, 1)
                                    .split('\n');

                                if (colorBy?.discover_score) {
                                    const id = context.dataset.data[context.dataIndex].point.id;
                                    const score = context.dataset.data[context.dataIndex].point.score;

                                    return [`id: ${id}`, `score: ${score}`];
                                } else {
                                    return payload;
                                }
                            },
                        },
                    },
                    legend: {
                        display: !!labelby,
                    },
                },
            },
            plugins: [
                {
                    id: 'myEventCatcher',
                    beforeEvent(chart, args) {
                        const event = args.event;
                        if (event.type === 'click') {
                            if (chart.tooltip._active.length > 0) {
                                const activePoints = chart.tooltip._active.map((point) => {
                                    return {
                                        id: point.element.$context.raw.point.id,
                                        payload: point.element.$context.raw.point.payload,
                                        vector: point.element.$context.raw.point.vector,
                                    };
                                });
                                setViewPoint(activePoints);
                                setOpenViewPoints(true);
                            }
                        }
                    },
                },
                {
                    id: 'AfterUpdate',
                    afterRender: () => {
                        channel.port1.postMessage(false);
                    }
                }
            ],
        });

        const worker = new Worker(new URL('./wasmWorker.js', import.meta.url), {
            type: 'module',
        });

        // if ((async () => await threads())()) {
        //   console.log('WebAssembly supported');
        //   worker = new Worker(new URL('./wasmWorker.js', import.meta.url), {
        //     type: 'module',
        //   });
        // }
        // else {
        //   console.log('WebAssembly not supported');
        //   worker = new Worker(new URL('./worker.js', import.meta.url), {
        //     type: 'module',
        //   });
        // }

        let sharedArray;
        let typedArray;
        let resultDataset;
        const outputDim = 2;

        worker.onmessage = (m) => {
            if (m.data.error) {
                enqueueSnackbar(`Visualization Unsuccessful\nError: ${m.data.error}`, {
                    variant: 'error',
                    style: { whiteSpace: 'pre-line' },
                    action,
                });
            }
            else if (m.data.error === null) {  // m.data.result && m.data.result.length > 0) {
                mutateDataset(resultDataset, typedArray, outputDim);
                resultDataset.forEach((dataset, index) => {
                    myChart.data.datasets[index].data = dataset.data;
                });
                myChart.update();
            }
            else {
                enqueueSnackbar(`Visualization Unsuccessful\nError: Unexpected Error Occured`, {
                    variant: 'error',
                    style: { whiteSpace: 'pre-line' },
                    action
                });
            }
        };

        // Error handling for worker
        worker.onerror = e => {
            console.error(e);
        }

        if (scrollResult.data.result?.points?.length > 0) {
            sharedArray = new SharedArrayBuffer(
                Float64Array.BYTES_PER_ELEMENT * scrollResult.data.result.points.length * outputDim
            );
            typedArray = new Float64Array(sharedArray);
            resultDataset = prepareDataset(scrollResult.data);
            worker.postMessage({ command: "CONN" }, [channel.port2]);
            worker.postMessage({
                command: "FWD",
                details: scrollResult.data,
                sharedArray,
                time: Date.now()
            });
        }

        return () => {
            myChart.destroy();
            worker.terminate();
            channel.port1.close();
            channel.port2.close();
        };
    }, [scrollResult]);

    return (
        <>
            <canvas id="myChart"></canvas>
            <ViewPointModal
                openViewPoints={openViewPoints}
                setOpenViewPoints={setOpenViewPoints}
                viewPoints={viewPoints}
            />
        </>
    );
};

VisualizeChart.propTypes = {
    scrollResult: PropTypes.object.isRequired,
};

export default VisualizeChart;
