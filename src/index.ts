import "@tensorflow/tfjs-backend-cpu";
import { promises as fs, existsSync } from "fs";
import * as jpeg from "jpeg-js";

// demo from https://github.com/tensorflow/tfjs-models/tree/master/mobilenet

// we commit this line because it's commonjs module
// const mobilenet = require('@tensorflow-models/mobilenet');
// in typescript we use this
import * as mobilenet from "@tensorflow-models/mobilenet";
import type { MobileNet } from "@tensorflow-models/mobilenet";

import * as canvas from "canvas";
import * as cv from "@techstark/opencv-js";
import { JSDOM } from "jsdom";

/**
 * wait
 */
function wait(sec: number) {
  return new Promise<void>((resolve) => {
    setTimeout(resolve, Math.floor(sec * 1000));
  });
}

/**
 * install dom into global object
 * because there is not dom in nodejs
 */
function installDOM(): void {
  const dom = new JSDOM();
  const anyGlobal = global as any;
  anyGlobal.document = dom.window.document;
  // The rest enables DOM image and canvas and is provided by node-canvas
  anyGlobal.Image = canvas.Image;
  anyGlobal.HTMLCanvasElement = canvas.Canvas;
  anyGlobal.ImageData = canvas.ImageData;
  anyGlobal.HTMLImageElement = canvas.Image;
  console.log("[js-dom] dom installed");
}

/**
 * wait until openCV is ready
 * @param openCvLib cv object from package
 * @returns promise
 */
function cvReady(openCvLib: typeof cv): Promise<void> {
  installDOM();
  return new Promise<void>((resolve) => {
    openCvLib.onRuntimeInitialized = () => {
      resolve();
      console.log("[OpenCV] runtime initialized");
    };
  });
}

let model: MobileNet | void;
let modelIsLoading = false;

/**
 * get mobilenet model
 * @returns mobilenet model
 */
async function getModel(): Promise<MobileNet> {
  if (model === undefined && !modelIsLoading) {
    modelIsLoading = true;
    model = await mobilenet.load();
    console.log("[TFJS] model loaded");
  }
  while (model === undefined) {
    await wait(1);
  }
  return model;
}

/**
 * openCV jobs for each file
 * @param srcDirname where examples are
 * @param distDirname where output directory should be
 * @param fileName filename
 */
async function doCvJobsForEachFile(
  srcDirname: string,
  distDirname: string,
  fileName: string
): Promise<void> {
  // read file
  const jpegData: Buffer = await fs.readFile(`${srcDirname}/${fileName}`);

  // decode it into ImageData
  const img: ImageData = jpeg.decode(jpegData) as unknown as ImageData;

  // Mat for openCV
  const mat = cv.matFromImageData(img);

  // a new Mat
  const newMat = mat.clone();

  // some random function from openCV
  cv.dilate(mat, newMat, new cv.Mat(5, 5, cv.CV_8UC1));

  // prepare a canvas
  const cnv = canvas.createCanvas(500, 500);

  // export Mat into canvas
  cv.imshow(cnv as unknown as HTMLCanvasElement, newMat);

  // get raw buffer data
  const buf = cnv.toBuffer();

  // write it into a file
  const outPath = `${distDirname}/${fileName}.dilated.png`;
  await fs.writeFile(outPath, buf);
  console.log(`[OpenCV] write file => "${outPath}"`);
}

/**
 * tensorflow jobs for each file
 * @param srcDirname where examples are
 * @param distDirname where output directory should be
 * @param fileName filename
 */
async function doTfJobsForEachFile(
  srcDirname: string,
  distDirname: string,
  fileName: string
): Promise<void> {
  // Load the model.
  const model = await getModel();

  // read file
  const jpegData: Buffer = await fs.readFile(`${srcDirname}/${fileName}`);

  // decode it into ImageData
  const img: ImageData = jpeg.decode(jpegData) as unknown as ImageData;

  // Classify the image.
  const predictions = await model.classify(img);

  // json string
  const jsonStr: string = JSON.stringify(predictions, undefined, 4);

  const buf: Buffer = Buffer.from(jsonStr, "utf-8");

  // write it into a file
  const outPath = `${distDirname}/${fileName}.prediction.json`;
  await fs.writeFile(outPath, buf);
  console.log(`[TFJS] write file => "${outPath}"`);
}

/**
 * main function
 */
async function main() {
  const srcDirname = "examples";
  const distDirname = "out";

  if (!existsSync(distDirname)) {
    await fs.mkdir(distDirname);
  }

  // get all names of image files under source directory
  const fileNames = await fs.readdir(srcDirname);

  await cvReady(cv);

  // loop for each images
  await Promise.all(
    fileNames
      .map((fileName) => [
        doTfJobsForEachFile(srcDirname, distDirname, fileName),
        doCvJobsForEachFile(srcDirname, distDirname, fileName),
      ])
      .reduce((res, jobs) => res.concat(jobs), [])
  );
}

// execute codes
main().catch(console.error);
