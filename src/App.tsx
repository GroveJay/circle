import { useCallback, useEffect, useRef, useState } from "react";
import cv from "@techstark/opencv-js";

const FOUR_PI = 4 * Math.PI;

const detectImageCircles = async (
  img: cv.Mat,
  gray: cv.Mat,
  dp: cv.double,
  min_dist: cv.double,
  param1: cv.double,
  param2: cv.double,
  min_radius: cv.int,
  max_radius: cv.int,
  loDiff: cv.Scalar,
  upDiff: cv.Scalar,
) => {
  cv.cvtColor(img, gray, cv.COLOR_RGBA2GRAY, 0);
  let circles = new cv.Mat();
  cv.HoughCircles(gray, circles, cv.HOUGH_GRADIENT, dp, min_dist, param1, param2, min_radius, max_radius);

  const circlesCount = circles.cols;
  let relevantContourCircularity = 0;

  if (circlesCount !== 1) {
    return {
      circlesCount,
      relevantContourCircularity,
    }
  }

  cv.medianBlur(gray, gray, 5)
  cv.adaptiveThreshold(gray, gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2);

  let mask = new cv.Mat();
  for (let i = 0; i < circles.cols; ++i) {
    let x = circles.data32F[i * 3];
    let y = circles.data32F[i * 3 + 1];
    let center = new cv.Point(x, y);
    cv.floodFill(gray, mask, center, new cv.Scalar(0), new cv.Rect(), loDiff, upDiff, cv.FLOODFILL_FIXED_RANGE); 
  }
  mask.delete();

  let contours = new cv.MatVector();
  let hierarchy = new cv.Mat();
  cv.findContours(gray, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_NONE);

  const maxArea = Math.PI * Math.pow(max_radius, 2);
  const onlyCircleCenter = new cv.Point(
    circles.data32F[0],
    circles.data32F[1]
  );
  
  for (let i = 0; i < (contours.size() as unknown as number); i++) {
    const contour = contours.get(i);
    if (cv.pointPolygonTest(contour, onlyCircleCenter, false) > 0) {
      const area = cv.contourArea(contour);
      if (area > maxArea) {
        continue;
      }
      const permiter = cv.arcLength(contour, true);
      const roundness = FOUR_PI * (area / Math.pow(permiter, 2));
      if (roundness > 0.5) {
        const contourMoments = cv.moments(contour);
        const centroidX = contourMoments['m10'] / contourMoments['m00'];
        const centroidY = contourMoments['m01'] / contourMoments['m00'];
        const contourDistances = [];
        let totalDistances = 0;
        for (var t = 0; t < contour.data32S.length; t+=2) {
            const x = contour.data32S[t];
            const y = contour.data32S[t+1];
            const distance = Math.sqrt(
                Math.pow(
                    x - centroidX,
                    2
                ) + Math.pow(
                    y - centroidY,
                    2
                )
            );
            totalDistances += distance;
            contourDistances.push(distance);
        }
        const averageDistance = totalDistances / contourDistances.length;
        console.log('averageDistance', averageDistance);
        let totalSquaredDistances = 0;
        for (var j = 0; j < contourDistances.length; j++) {
            totalSquaredDistances += Math.pow((contourDistances[j] - averageDistance), 2);
        }
        console.log('totalSquaredDistances', totalSquaredDistances);
        const sd = Math.sqrt(totalSquaredDistances / contourDistances.length);
        console.log('sd', sd);
        const circularity = sd / averageDistance;
        console.log(circularity);
        relevantContourCircularity = 1 - circularity;
        break;
      }
    }
  }

  contours.delete();
  hierarchy.delete();
  circles.delete();
  return {
    circlesCount,
    relevantContourCircularity,
  }
}

function App() {
  const [initialized, setInitialized] = useState(false);
  const [paused, setPaused] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [circles, setCircles] = useState(0);
  const [circularity, setCircularity] = useState(0);
  const [maxRoundnessSeen, setMaxRoundnessSeen] = useState(0);
  const [neighborDistance, setNeighborDistance] = useState(250);
  const [param1,setParam1] = useState(40);
  const [param2, setParam2] = useState(35);
  const [minRadius,setMinRadius] = useState(95);
  const [maxRadius,setMaxRadius] = useState(145);
  const [loDiff, setLoDiff] = useState(0);
  const [upDiff, setUpDiff] = useState(0);

  const videoInputRef = useRef<HTMLVideoElement>(null);
  const circledImageRef = useRef<HTMLCanvasElement>(null);
  const source = useRef<cv.Mat>(null);
  const gray = useRef<cv.Mat>(null);

  const capture = useRef<cv.VideoCapture>(null);

  const detectCircles = useCallback(async () => {
    if (!capture.current) {
      console.warn('no capture element!');
      return;
    }
    if (!source.current) {
      console.warn('no source element!');
      return;
    }
    capture.current.read(source.current);
    
    const { circlesCount, relevantContourCircularity } = await detectImageCircles(
      source.current, gray.current, 1, neighborDistance, param1, param2, minRadius, maxRadius,
      new cv.Scalar(loDiff), new cv.Scalar(upDiff),
    );
    setCircles(circlesCount);
    setCircularity(relevantContourCircularity);
    setMaxRoundnessSeen((previousMaxRoundnessSeen) => {
      if (relevantContourCircularity > previousMaxRoundnessSeen) {
        return relevantContourCircularity;
      }
      return previousMaxRoundnessSeen;
    });

    cv.circle(gray.current, new cv.Point(circledImageRef.current.width / 2, circledImageRef.current.height / 2), maxRadius, new cv.Scalar(128), 3);
    try {
      cv.imshow(circledImageRef.current, gray.current);
    } catch (e) {
      console.warn('caught exception: ', e);
    }
    
  }, [neighborDistance, param1, param2, minRadius, maxRadius]);

  const initialize = useCallback(async () => {
    if (!videoInputRef || !videoInputRef.current || initialized) return;
    let stream = undefined;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: {
            exact: "environment",
          },
          width: { min: 1280, max: 1560 },
          height: { min: 720, max: 1440 }
        },
        audio: false,
      });
    } catch {
      stream = await navigator.mediaDevices.getUserMedia(
        {
          video: true,/*{
             width: { min: 1280, max: 1560 },
             height: { min: 720, max: 1440 }
          },*/
          audio: false,
        });
    }
    if (!stream) {
      console.log("An error occurred!");
      return
    }
    videoInputRef.current.srcObject = stream;
    await videoInputRef.current.play();

    videoInputRef.current.height = videoInputRef.current.videoHeight;
    videoInputRef.current.width = videoInputRef.current.videoWidth;
    const height = videoInputRef.current.videoHeight;
    const width = videoInputRef.current.videoWidth;
    source.current = new cv.Mat(height, width, cv.CV_8UC4);
    gray.current = new cv.Mat();
    capture.current = new cv.VideoCapture(videoInputRef.current)

    const smallerDimension = Math.min(width, height);
    setMinRadius((smallerDimension * .6) / 2);
    setMaxRadius((smallerDimension * .8) / 2);
    setNeighborDistance(smallerDimension / 2);
    setInitialized(true);
  }, [videoInputRef.current, initialized]);

  useEffect(() => {
    if (cv.getBuildInformation) { 
      initialize();
    } else {
      cv['onRuntimeInitialized']=()=>{
        initialize();
      }
    }
  }, [initialize]);

  useEffect(() => {
    if (!initialized || paused) return;
    let handle: number;
    const nextTick = () => {
      handle = requestAnimationFrame(async () => {
        await detectCircles();
        nextTick();
      });
    };
    nextTick();
    return () => {
      cancelAnimationFrame(handle);
    };
  }, [initialized, detectCircles, paused]);

  return (
    <div className="App">
      <video className="videoInput" ref={videoInputRef} />
      <canvas className="outputImage" ref={circledImageRef} />
      <div>
        <h4
          style={{ margin: '2px 0' }}
        >
          Circularity: {(circularity * 100).toFixed(2)} % (max: {(maxRoundnessSeen * 100).toFixed(2)} %)
        </h4>
      </div>
      <div className="settings">
        <div>
          <button
            onClick={() => {
              setMaxRoundnessSeen(0);
            }}
          >
            Reset Max Roundness
          </button>
          <button
            onClick={() => {
              setShowSettings(!showSettings);
            }}
          >
            ?
          </button>
        </div>
        <div style={{
          display: showSettings ? '' : 'none',
          flexDirection: 'row',
          maxWidth: '600px',
          margin: '0 auto',
          flexWrap: 'wrap',
          justifyContent: 'space-evenly'
        }}>
          <button
            onClick={() => {
              setPaused(!paused);
            }}
          >
            {paused ? 'Unpause' : 'Pause'}
          </button>
          <InputSlider
            description="Neighbor Distance"
            value={neighborDistance}
            min={1}
            max={1000}
            set={setNeighborDistance}
          />
          <InputSlider
            description="Param 1"
            value={param1}
            min={1}
            max={200}
            set={setParam1}
          />
          <InputSlider
            description="Param 2"
            value={param2}
            min={1}
            max={200}
            set={setParam2}
          />
          <InputSlider
            description="Min Circle Radius"
            value={minRadius}
            min={25}
            max={maxRadius}
            set={setMinRadius}
          />
          <InputSlider
            description="Max Circle Radius"
            value={maxRadius}
            min={minRadius}
            max={250}
            set={setMaxRadius}
          />
          <InputSlider
            description="loDiff"
            value={loDiff}
            min={0}
            max={255}
            set={setLoDiff}
          />
          <InputSlider
            description="upDiff"
            value={upDiff}
            min={0}
            max={255}
            set={setUpDiff}
          />
        </div>
      </div>
    </div>
  );
}

export default App

const InputSlider: React.FC<{
  description: string;
  value: number;
  min: number;
  max: number;
  set: (value: React.SetStateAction<number>) => void
}> = ({
  description,
  value,
  min,
  max,
  set,
}) => {
  return (
    <div className="inputSlider">
      <p><u>{description}</u>: {value}</p>
      <input
        type="range"
        min={min}
        max={max}
        value={value}
        onChange={(e) => { set(parseInt(e.currentTarget.value)) }}
      />
    </div>
  )
}