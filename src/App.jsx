import { useCallback, useEffect, useRef, useState } from 'react'
import labelMap from './data/msl-label-map.json'
import annotationText from './data/msl-annotations.txt?raw'
import './App.css'

const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
const POSE_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task'
const MODEL_SOURCE_REVISION = '76d200c5b3eea3a13c4f314e61334b5d725611b5'
const MODEL_SOURCE = `https://raw.githubusercontent.com/lawun330/myanmar-sign-language-recognition/${MODEL_SOURCE_REVISION}/onnx_models`
const ONNX_MODEL_URL = `${MODEL_SOURCE}/bilstm_msl.onnx`
const ONNX_WEIGHTS_URL = `${MODEL_SOURCE}/bilstm_msl.onnx.data`
const MAX_SEQUENCE_LENGTH = 200
const LANDMARK_COUNT = 75
const COORDINATE_COUNT = 3
const FEATURE_COUNT = LANDMARK_COUNT * COORDINATE_COUNT
const LABELS_BY_INDEX = Object.entries(labelMap.label2idx)
  .sort(([, first], [, second]) => first - second)
  .map(([label]) => label)
const GLOSS_TO_WORD = new Map(annotationText
  .split(/\r?\n/)
  .filter(Boolean)
  .map((line) => line.split('\t'))
  .filter(([written, gloss]) => written && gloss)
  .map(([written, gloss]) => [gloss.trim(), written.replace(/\s*။\s*$/, '').trim()]))

const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
]

function HandShape({ hands, label }) {
  const visibleHands = hands || []
  const visiblePoints = visibleHands.flatMap((points) => points.filter(([x, y]) => Math.abs(x) + Math.abs(y) > 1e-6))
  if (!visiblePoints.length) return <span className="shape-placeholder" aria-hidden="true">✳</span>

  const minX = Math.min(...visiblePoints.map(([x]) => x))
  const maxX = Math.max(...visiblePoints.map(([x]) => x))
  const minY = Math.min(...visiblePoints.map(([, y]) => y))
  const maxY = Math.max(...visiblePoints.map(([, y]) => y))
  const width = Math.max(maxX - minX, 0.001)
  const height = Math.max(maxY - minY, 0.001)
  const scale = Math.min(104 / width, 48 / height)
  const offsetX = (120 - width * scale) / 2 - minX * scale
  const offsetY = (58 - height * scale) / 2 - minY * scale + 3
  const project = ([x, y]) => [x * scale + offsetX, y * scale + offsetY]

  return <svg className="hand-shape-svg" viewBox="0 0 120 64" role="img" aria-label={`Hand shape for ${label}`}>
    {visibleHands.map((points, handIndex) => {
      const projected = points.map(project)
      const isVisible = points.map(([x, y]) => Math.abs(x) + Math.abs(y) > 1e-6)
      const tone = handIndex % 2 === 0 ? 'mint' : 'gold'
      return <g className={`shape-hand ${tone}`} key={handIndex}>
        {CONNECTIONS.map(([from, to]) => isVisible[from] && isVisible[to] && <line key={`${from}-${to}`} x1={projected[from][0]} y1={projected[from][1]} x2={projected[to][0]} y2={projected[to][1]} />)}
        {projected.map(([x, y], index) => isVisible[index] && <circle key={index} cx={x} cy={y} r={index === 0 || [4, 8, 12, 16, 20].includes(index) ? 1.9 : 1.2} />)}
      </g>
    })}
  </svg>
}

function makeKeypointFrame(poseResult, handResult) {
  const coordinates = new Float32Array(FEATURE_COUNT)
  const pose = poseResult.landmarks?.[0] || []
  pose.slice(0, 33).forEach((point, index) => {
    const start = index * COORDINATE_COUNT
    coordinates[start] = point.x
    coordinates[start + 1] = point.y
    coordinates[start + 2] = point.z
  })

  const detectedHands = handResult.landmarks || []
  detectedHands.forEach((landmarks, index) => {
    const handedness = handResult.handedness?.[index]?.[0]?.categoryName
    const handStart = handedness === 'Right' ? 33 : 54
    landmarks.slice(0, 21).forEach((point, pointIndex) => {
      const start = (handStart + pointIndex) * COORDINATE_COUNT
      coordinates[start] = point.x
      coordinates[start + 1] = point.y
      coordinates[start + 2] = point.z
    })
  })

  return { coordinates, hasHands: detectedHands.length > 0 }
}

function makeClassifierTensors(samples, ort) {
  const selected = samples.length > MAX_SEQUENCE_LENGTH
    ? samples.slice(Math.floor((samples.length - MAX_SEQUENCE_LENGTH) / 2), Math.floor((samples.length - MAX_SEQUENCE_LENGTH) / 2) + MAX_SEQUENCE_LENGTH)
    : samples
  const featureData = new Float32Array(MAX_SEQUENCE_LENGTH * FEATURE_COUNT)
  const paddingMask = new Uint8Array(MAX_SEQUENCE_LENGTH)
  const sequenceLength = selected.length

  selected.forEach((sample, frameIndex) => {
    const frame = sample.coordinates
    const leftShoulder = 11 * COORDINATE_COUNT
    const rightShoulder = 12 * COORDINATE_COUNT
    const midpoint = [0, 1, 2].map((axis) => (frame[leftShoulder + axis] + frame[rightShoulder + axis]) / 2)
    const shoulderWidth = Math.hypot(
      frame[rightShoulder] - frame[leftShoulder],
      frame[rightShoulder + 1] - frame[leftShoulder + 1],
      frame[rightShoulder + 2] - frame[leftShoulder + 2],
    ) || 1

    for (let featureIndex = 0; featureIndex < FEATURE_COUNT; featureIndex += 1) {
      const axis = featureIndex % COORDINATE_COUNT
      const normalized = (frame[featureIndex] - midpoint[axis]) / shoulderWidth
      featureData[frameIndex * FEATURE_COUNT + featureIndex] = axis === 2 ? 0 : normalized
    }
  })

  paddingMask.fill(1, sequenceLength)
  return {
    keypoints: new ort.Tensor('float32', featureData, [1, MAX_SEQUENCE_LENGTH, FEATURE_COUNT]),
    mask: new ort.Tensor('bool', paddingMask, [1, MAX_SEQUENCE_LENGTH]),
  }
}

export default function App() {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const landmarkerRef = useRef(null)
  const poseLandmarkerRef = useRef(null)
  const classifierRef = useRef(null)
  const frameRef = useRef(0)
  const facingModeRef = useRef('user')
  const ortModuleRef = useRef(null)
  const keypointBufferRef = useRef([])
  const predictionBusyRef = useRef(false)
  const predictionVotesRef = useRef([])
  const lastPredictionAtRef = useRef(0)
  const [cameraOn, setCameraOn] = useState(false)
  const [loading, setLoading] = useState(false)
  const [modelStatus, setModelStatus] = useState('')
  const [error, setError] = useState('')
  const [hands, setHands] = useState([])
  const [fps, setFps] = useState(0)
  const [facingMode, setFacingMode] = useState('user')
  const [topPredictions, setTopPredictions] = useState([])
  const [recognizedWords, setRecognizedWords] = useState([])
  const [wordShapes, setWordShapes] = useState({})
  const [wordSearch, setWordSearch] = useState('')
  const [wordPage, setWordPage] = useState(0)

  useEffect(() => {
    let active = true
    fetch(`${import.meta.env.BASE_URL}data/msl-hand-shapes.json`)
      .then((response) => {
        if (!response.ok) throw new Error('Could not load sign illustrations')
        return response.json()
      })
      .then((shapes) => { if (active) setWordShapes(shapes) })
      .catch(() => {})
    return () => { active = false }
  }, [])

  const stopCamera = useCallback(() => {
    cancelAnimationFrame(frameRef.current)
    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach((track) => track.stop())
      videoRef.current.srcObject = null
    }
    const canvas = canvasRef.current
    if (canvas) canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height)
    setCameraOn(false)
    setHands([])
    setFps(0)
    keypointBufferRef.current = []
    predictionVotesRef.current = []
    predictionBusyRef.current = false
    setTopPredictions([])
    setRecognizedWords([])
  }, [])

  const initializeModels = useCallback(async () => {
    if (landmarkerRef.current && poseLandmarkerRef.current && classifierRef.current) return
    setModelStatus('Loading sign recognition models…')
    const { FilesetResolver, HandLandmarker, PoseLandmarker } = await import('@mediapipe/tasks-vision')
    const ort = await import('onnxruntime-web/wasm')
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/'
    ort.env.wasm.numThreads = 1
    ortModuleRef.current = ort
    const vision = await FilesetResolver.forVisionTasks(`${import.meta.env.BASE_URL}mediapipe/wasm`)
    const createVisionTask = async (Task, taskOptions) => {
      try {
        return await Task.createFromOptions(vision, {
          ...taskOptions,
          baseOptions: { ...taskOptions.baseOptions, delegate: 'GPU' },
        })
      } catch {
        return Task.createFromOptions(vision, {
          ...taskOptions,
          baseOptions: { ...taskOptions.baseOptions, delegate: 'CPU' },
        })
      }
    }
    const [handTask, poseTask, classifier] = await Promise.all([
      createVisionTask(HandLandmarker, {
        baseOptions: { modelAssetPath: MODEL_URL },
        runningMode: 'VIDEO',
        numHands: 2,
        minHandDetectionConfidence: 0.55,
        minHandPresenceConfidence: 0.55,
        minTrackingConfidence: 0.5,
      }),
      createVisionTask(PoseLandmarker, {
        baseOptions: { modelAssetPath: POSE_MODEL_URL },
        runningMode: 'VIDEO',
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      }),
      ort.InferenceSession.create(ONNX_MODEL_URL, {
        executionProviders: ['wasm'],
        externalData: [{ path: 'bilstm_msl.onnx.data', data: ONNX_WEIGHTS_URL }],
      }),
    ])
    landmarkerRef.current = handTask
    poseLandmarkerRef.current = poseTask
    classifierRef.current = classifier
    setModelStatus('Models ready · Myanmar Sign Language')
  }, [])

  const classifySequence = useCallback(async (samples) => {
    const classifier = classifierRef.current
    if (!classifier || predictionBusyRef.current) return
    predictionBusyRef.current = true
    try {
      const tensors = makeClassifierTensors(samples, ortModuleRef.current)
      const output = await classifier.run(tensors)
      const logits = output[classifier.outputNames[0]].data
      const maxLogit = Math.max(...logits)
      const exponentials = Array.from(logits, (value) => Math.exp(value - maxLogit))
      const denominator = exponentials.reduce((sum, value) => sum + value, 0)
      const top = exponentials
        .map((value, index) => ({ index, confidence: value / denominator }))
        .sort((first, second) => second.confidence - first.confidence)
        .slice(0, 5)
        .map(({ index, confidence }) => ({
          label: LABELS_BY_INDEX[index] || `Class ${index}`,
          word: GLOSS_TO_WORD.get(LABELS_BY_INDEX[index]) || LABELS_BY_INDEX[index] || `Class ${index}`,
          confidence,
        }))

      setTopPredictions(top)
      if (top[0]?.confidence >= 0.15) {
        const now = Date.now()
        predictionVotesRef.current = [...predictionVotesRef.current, { ...top[0], at: now }].slice(-3)
        const voteCount = predictionVotesRef.current.filter((vote) => vote.label === top[0].label).length
        if (voteCount >= 2) {
          setRecognizedWords((current) => {
            if (current[0]?.label === top[0].label) return current
            return [{ ...top[0], at: now }, ...current].slice(0, 8)
          })
        }
      }
    } catch (predictionError) {
      setError(`Sign recognition stopped: ${predictionError.message || 'model inference failed'}`)
    } finally {
      predictionBusyRef.current = false
    }
  }, [])

  const startCamera = useCallback(async () => {
    setError('')
    setLoading(true)
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera access requires a secure connection (HTTPS or localhost).')
      await initializeModels()
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: facingModeRef.current, width: { ideal: 1280 }, height: { ideal: 800 } },
      })
      const video = videoRef.current
      video.srcObject = stream
      await video.play()
      setCameraOn(true)
    } catch (err) {
      setError(err.name === 'NotAllowedError'
        ? 'Camera permission was blocked. Allow camera access in your browser and try again.'
        : err.message || 'Could not start the camera. Please try again.')
      stopCamera()
    } finally {
      setLoading(false)
    }
  }, [initializeModels, stopCamera])

  useEffect(() => {
    if (!cameraOn) return undefined
    let lastTime = 0
    let fpsCount = 0
    let fpsStart = performance.now()
    const render = () => {
      const video = videoRef.current
      const canvas = canvasRef.current
      const detector = landmarkerRef.current
      const poseDetector = poseLandmarkerRef.current
      if (!video || !canvas || !detector || !poseDetector || video.readyState < 2) {
        frameRef.current = requestAnimationFrame(render)
        return
      }
      const width = video.videoWidth
      const height = video.videoHeight
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width
        canvas.height = height
      }
      const now = performance.now()
      if (now !== lastTime) {
        const result = detector.detectForVideo(video, now)
        const poseResult = poseDetector.detectForVideo(video, now)
        const detectedHands = result.landmarks || []
        const sample = makeKeypointFrame(poseResult, result)
        keypointBufferRef.current = [...keypointBufferRef.current.slice(-99), sample]
        setHands(detectedHands.map((_, index) => result.handedness?.[index]?.[0]?.categoryName || 'Hand'))
        const ctx = canvas.getContext('2d')
        ctx.clearRect(0, 0, width, height)
        ctx.save()
        ctx.translate(width, 0)
        ctx.scale(-1, 1)
        detectedHands.forEach((landmarks) => {
          ctx.lineCap = 'round'
          ctx.lineJoin = 'round'
          ctx.strokeStyle = 'rgba(98, 236, 197, .84)'
          ctx.lineWidth = Math.max(3, width / 280)
          CONNECTIONS.forEach(([from, to]) => {
            ctx.beginPath()
            ctx.moveTo(landmarks[from].x * width, landmarks[from].y * height)
            ctx.lineTo(landmarks[to].x * width, landmarks[to].y * height)
            ctx.stroke()
          })
          landmarks.forEach((point, index) => {
            ctx.beginPath()
            ctx.arc(point.x * width, point.y * height, index % 4 === 0 ? width / 150 : width / 260, 0, Math.PI * 2)
            ctx.fillStyle = index % 4 === 0 ? '#f7c76a' : '#f2fbff'
            ctx.fill()
          })
        })
        ctx.restore()
        lastTime = now
        fpsCount += 1
        const recentSamples = keypointBufferRef.current.slice(-60)
        const handsInWindow = recentSamples.filter((item) => item.hasHands).length
        if (now - lastPredictionAtRef.current > 1200
          && keypointBufferRef.current.length >= 60
          && handsInWindow >= 12
          && !predictionBusyRef.current) {
          lastPredictionAtRef.current = now
          void classifySequence([...keypointBufferRef.current])
        }
        if (now - fpsStart > 800) {
          setFps(Math.round((fpsCount * 1000) / (now - fpsStart)))
          fpsCount = 0
          fpsStart = now
        }
      }
      frameRef.current = requestAnimationFrame(render)
    }
    frameRef.current = requestAnimationFrame(render)
    return () => cancelAnimationFrame(frameRef.current)
  }, [cameraOn, classifySequence])

  useEffect(() => () => {
    if (videoRef.current?.srcObject) videoRef.current.srcObject.getTracks().forEach((track) => track.stop())
    landmarkerRef.current?.close()
    poseLandmarkerRef.current?.close()
    classifierRef.current?.release()
  }, [])

  const switchCamera = () => {
    const nextFacing = facingMode === 'user' ? 'environment' : 'user'
    facingModeRef.current = nextFacing
    setFacingMode(nextFacing)
    if (cameraOn) {
      stopCamera()
      window.setTimeout(() => startCamera(), 0)
    }
  }

  const vocabulary = LABELS_BY_INDEX.map((label, index) => ({
    index,
    label,
    word: GLOSS_TO_WORD.get(label) || label,
  }))
  const filteredVocabulary = vocabulary.filter(({ label, word }) => (
    `${label} ${word}`.toLocaleLowerCase().includes(wordSearch.trim().toLocaleLowerCase())
  ))
  const vocabularyPageSize = 10
  const vocabularyPageCount = Math.max(1, Math.ceil(filteredVocabulary.length / vocabularyPageSize))
  const visibleVocabulary = filteredVocabulary.slice(
    wordPage * vocabularyPageSize,
    (wordPage + 1) * vocabularyPageSize,
  )

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Signa home">
          <span className="brand-mark"><span /><span /><span /><span /></span>
          <span>signa<span className="brand-period">.</span></span>
        </a>
        <div className="topbar-right"><span className="topbar-label">INTERACTIVE DEMO</span><span className="topbar-divider" /><span className="model-chip"><span className="chip-dot" /> MEDIAPIPE</span></div>
      </header>

      <section className="hero" id="top">
        <div className="hero-copy">
          <div className="eyebrow"><span className="eyebrow-line" /> COMPUTER VISION, IN REAL TIME</div>
          <h1>Every gesture<br />has a <span>voice.</span></h1>
          <p className="hero-subtitle">Myanmar Sign Language Recognition</p>
          <p className="hero-description">Track your hands and body in real time, then see the Myanmar sign words predicted from each movement sequence.</p>
          <div className="hero-meta"><span><span className="meta-icon">✳</span> Runs right in your browser</span><span><span className="meta-icon">◉</span> Nothing leaves your device</span></div>
        </div>

        <div className="demo-card">
          <div className="card-topline"><div className="live-label"><span className={`live-dot ${cameraOn ? 'active' : ''}`} /> {cameraOn ? 'LIVE TRACKING' : 'CAMERA PREVIEW'}</div><div className="card-top-right"><span className="fps-label">{cameraOn ? `${fps} FPS` : 'READY'}</span><span className="menu-dots">···</span></div></div>
          <div className={`camera-stage ${cameraOn ? 'is-live' : ''}`}>
            <video ref={videoRef} className="camera-video" playsInline muted aria-label="Live camera preview" />
            <canvas ref={canvasRef} className="landmark-canvas" />
            {!cameraOn && <div className="camera-placeholder"><div className="scan-frame"><i className="corner tl" /><i className="corner tr" /><i className="corner bl" /><i className="corner br" /><div className="hand-illustration"><svg viewBox="0 0 120 140" role="img" aria-label="Hand outline"><path d="M34 69V35a7 7 0 0 1 14 0v28-43a7 7 0 0 1 14 0v42-37a7 7 0 0 1 14 0v40-28a7 7 0 0 1 14 0v52c0 27-15 42-39 42h-2c-15 0-26-8-34-23L9 84a8 8 0 0 1 14-8l11 15V69Z" /></svg><span className="orbit orbit-one" /><span className="orbit orbit-two" /><span className="crosshair crosshair-one">+</span><span className="crosshair crosshair-two">+</span></div><div className="placeholder-text"><span className="placeholder-title">Your camera, your canvas</span><span className="placeholder-subtitle">Start the camera to begin tracking</span></div></div></div>}
            {cameraOn && <div className="tracking-badge"><span className="tracking-pulse" /> TRACKING ACTIVE</div>}
            <div className="stage-corner-label">{cameraOn ? `${hands.length} ${hands.length === 1 ? 'HAND' : 'HANDS'} IN FRAME` : 'AWAITING CAMERA'}</div>
          </div>
          <div className="card-controls"><div className="detection-status"><span className={`status-indicator ${cameraOn ? 'on' : ''}`} /><span>{cameraOn ? (hands.length ? `${hands.length} hand${hands.length > 1 ? 's' : ''} detected` : 'Looking for hands…') : 'Camera is off'}</span></div><div className="control-buttons"><button className="icon-button" type="button" onClick={switchCamera} aria-label="Switch camera" title="Switch camera">⇄</button><button className={`camera-button ${cameraOn ? 'stop-button' : ''}`} type="button" onClick={cameraOn ? stopCamera : startCamera} disabled={loading}>{loading ? <><span className="spinner" /> Loading models…</> : cameraOn ? <><span className="stop-icon" /> Stop camera</> : <><span className="camera-icon" /> Start camera</>}</button></div></div>
          {error && <div className="error-message" role="alert">{error}</div>}
          <section className="word-dashboard" aria-label="Myanmar sign recognition dashboard">
            <div className="dashboard-heading">
              <div>
                <span className="dashboard-kicker">LIVE RECOGNITION</span>
                <h2>Sign-to-word dashboard</h2>
                <p>Myanmar emergency signs · trained from the local keypoint dataset</p>
              </div>
              <span className={`model-status ${cameraOn ? 'is-ready' : ''}`}><span />{cameraOn ? modelStatus || 'Recognition active' : 'Start tracking to recognize signs'}</span>
            </div>

            <div className="recognition-grid">
              <div className="latest-match" aria-live="polite">
                <span className="output-caption">TOP LIVE MATCH</span>
                {topPredictions[0] ? <>
                  <strong lang="my">{topPredictions[0].word}</strong>
                  <span className="match-gloss" lang="my">Sign label · {topPredictions[0].label}</span>
                  <span className="match-confidence">{Math.round(topPredictions[0].confidence * 100)}% confidence</span>
                </> : <>
                  <strong className="waiting-word" lang="my">စတင်ရန်</strong>
                  <span className="match-gloss">Start a session and sign in view</span>
                  <span className="match-confidence">Waiting for a sign sequence</span>
                </>}
              </div>
              <div className="session-transcript">
                <div className="transcript-heading"><div><span className="output-caption">SESSION WORDS</span><strong>Recognized signs</strong></div><span className="transcript-count">{recognizedWords.length.toString().padStart(2, '0')}</span></div>
                {recognizedWords.length ? <ol className="transcript-list">{recognizedWords.map((item, index) => <li key={`${item.label}-${item.at}`}><span className="transcript-index">{String(recognizedWords.length - index).padStart(2, '0')}</span><span lang="my">{item.word}</span><small>{Math.round(item.confidence * 100)}%</small></li>)}</ol> : <p className="transcript-empty">Stable predictions will appear here as you sign.</p>}
              </div>
            </div>

            <div className="vocabulary-heading">
              <div><span className="output-caption">FROM YOUR DATASET</span><h3>Word library</h3></div>
              <label className="word-search"><span>⌕</span><input value={wordSearch} onChange={(event) => { setWordSearch(event.target.value); setWordPage(0) }} placeholder="Search Myanmar signs" aria-label="Search dataset vocabulary" /></label>
            </div>
            <div className="word-library" aria-live="polite">
              {visibleVocabulary.length ? visibleVocabulary.map(({ index, label, word }) => <div className="word-tile" key={label}>
                <span className="tile-index">{String(index + 1).padStart(3, '0')}</span>
                <div className="word-shape"><HandShape hands={wordShapes[label]} label={word} /></div>
                <strong lang="my">{word}</strong>
                <small lang="my">{label !== word ? label : 'Myanmar sign'}</small>
              </div>) : <p className="no-words">No vocabulary entries match “{wordSearch}”.</p>}
            </div>
            <div className="library-footer"><span>{filteredVocabulary.length} words from the dataset</span><div className="page-controls"><button type="button" onClick={() => setWordPage((page) => Math.max(0, page - 1))} disabled={wordPage === 0} aria-label="Previous words">←</button><span>{wordPage + 1} / {vocabularyPageCount}</span><button type="button" onClick={() => setWordPage((page) => Math.min(vocabularyPageCount - 1, page + 1))} disabled={wordPage >= vocabularyPageCount - 1} aria-label="Next words">→</button></div></div>
          </section>
        </div>
      </section>

      <section className="details-row" aria-label="Tracking details"><div className="detail-item"><span className="detail-number">01</span><div><span className="detail-title">21 LANDMARKS</span><span className="detail-copy">Every joint, mapped</span></div></div><span className="detail-separator" /><div className="detail-item"><span className="detail-number">02</span><div><span className="detail-title">2 HANDS</span><span className="detail-copy">Tracked simultaneously</span></div></div><span className="detail-separator" /><div className="detail-item"><span className="detail-number">03</span><div><span className="detail-title">ON-DEVICE</span><span className="detail-copy">Private by design</span></div></div><div className="powered-by"><span className="powered-line" /> POWERED BY <strong>MEDIAPIPE</strong></div></section>
      <footer className="footer"><span>MADE FOR A MORE CONNECTED WORLD</span><span>HAND LANDMARKER <b>·</b> WEB DEMO</span></footer>
    </main>
  )
}
