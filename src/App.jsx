import { useCallback, useEffect, useRef, useState } from 'react'
import labelMap from './data/msl-label-map.json'
import annotationText from './data/msl-annotations.txt?raw'
import './App.css'

const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, '') || 'https://sign-translate-api-production.up.railway.app'
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

const SIGN_LESSONS = [
  { english: 'Hello', myanmar: 'မင်္ဂလာပါ', shape: 'point', pose: 'forehead', other: 'Rest the other hand near your abdomen.', movement: 'The source identifies this as a dynamic sign. Its figure shows the index finger pointing toward the face near the eye or forehead; follow a fluent signer for the complete motion.', face: 'Friendly greeting.', sourceNote: 'Figure 1(a) in the MSL study.', motion: '' },
  { english: 'How are you?', myanmar: 'နေကောင်းလား', shape: 'point', pose: 'forehead', other: 'The other hand begins near the abdomen.', movement: 'Begin with one hand near the forehead, then move it down toward the chest.', face: 'Facial expression matters: this is a question.', sourceNote: 'Figure 3(a) in the MSL study.', motion: 'down' },
  { english: "It's okay", myanmar: 'အဆင်ပြေတယ်', shape: 'both', pose: 'chest', other: 'Use both hands together in front of your chest.', movement: 'Start with both hands close together at chest level, then move them apart.', face: 'Facial expression matters to distinguish the meaning.', sourceNote: 'Figure 3(b) in the MSL study.', motion: 'apart' },
  { english: 'Monday', myanmar: 'တနင်္လာ', shape: 'point', pose: 'eye', other: 'Rest the other hand near your abdomen.', movement: 'Uses the same hand sign as Hello, but at a different position near the nose or eye area.', face: 'Keep the expression natural.', sourceNote: 'Figure 1(b) in the MSL study.' },
  { english: 'Tuesday', myanmar: 'အင်္ဂါ', shape: 'point', pose: 'unknown', other: 'The source does not specify the exact position in its text description.', movement: 'The study says Tuesday shares a hand sign with Sunday, with a different position. Compare the two lessons with a fluent signer.', face: 'Use the expression shown by a fluent signer.', sourceNote: 'The study describes the Tuesday/Sunday comparison.' },
  { english: 'Wednesday', myanmar: 'ဗုဒ္ဓဟူး', shape: 'unknown', pose: 'unknown', other: 'The source does not specify the exact hand position in its text description.', movement: 'Wednesday and Thursday use the same hand position but different hand signs. Learn their exact shapes from a fluent signer.', face: 'Use the expression shown by a fluent signer.', sourceNote: 'The study describes the Wednesday/Thursday comparison.' },
  { english: 'Thursday', myanmar: 'ကြာသပတေး', shape: 'unknown', pose: 'unknown', other: 'The source does not specify the exact hand position in its text description.', movement: 'Thursday and Wednesday use the same hand position but different hand signs. Learn their exact shapes from a fluent signer.', face: 'Use the expression shown by a fluent signer.', sourceNote: 'The study describes the Wednesday/Thursday comparison.' },
  { english: 'Friday', myanmar: 'သောကြာ', shape: 'close', pose: 'mouth', other: 'The other hand rests near the abdomen.', movement: 'Hold one hand near the mouth or cheek, with fingers close to the face.', face: 'Keep the expression natural.', sourceNote: 'Figure 2(a) in the MSL study.' },
  { english: 'Saturday', myanmar: 'စနေ', shape: 'close', pose: 'mouth', other: 'The other hand rests near the abdomen.', movement: 'Uses a different hand sign from Friday at a similar mouth or cheek position, with fingers close to the face.', face: 'Keep the expression natural.', sourceNote: 'Figure 2(b) in the MSL study.' },
  { english: 'Sunday', myanmar: 'တနင်္ဂနွေ', shape: 'point', pose: 'unknown', other: 'The source does not specify the exact position in its text description.', movement: 'The study says Sunday shares a hand sign with Tuesday, with a different position. Compare the two lessons with a fluent signer.', face: 'Use the expression shown by a fluent signer.', sourceNote: 'The study describes the Tuesday/Sunday comparison.' },
]

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

function HandPlacementGuide() {
  return <div className="hand-placement-guide" aria-label="Hand position guide">
    <div className="guide-copy">
      <span>HAND SHAPE GUIDE</span>
      <small>Face your palm toward the camera. Keep your wrist and fingertips in frame.</small>
    </div>
    <svg viewBox="0 0 160 180" aria-hidden="true">
      <rect className="guide-boundary" x="12" y="8" width="136" height="164" rx="8" />
      <path className="guide-palm" d="M53 87V47c0-8 12-8 12 0v31V31c0-9 13-9 13 0v46V25c0-9 13-9 13 0v53V38c0-9 13-9 13 0v62c0 32-19 51-46 51-18 0-31-10-39-27l-11-23c-4-9 8-15 13-7l12 17V87c0-8 12-8 12 0Z" />
      <path className="guide-bone" d="M59 140V88m0-10V47m13 30V31m13 46V25m13 52V38m-39 50 13 2 13-2 13 1m-39 17 13 2 13-2 13 1m-39 16 13 2 13-2 13 1m-39 15 13 2 13-2" />
      <path className="guide-axis" d="M80 12v156" />
      <circle className="guide-joint" cx="59" cy="47" r="3"/><circle className="guide-joint" cx="72" cy="31" r="3"/><circle className="guide-joint" cx="85" cy="25" r="3"/><circle className="guide-joint" cx="98" cy="38" r="3"/>
    </svg>
  </div>
}

function LessonHandIcon({ pose, shape, movement }) {
  const handPoint = pose === 'forehead' || pose === 'eye'
  const handMouth = pose === 'mouth'
  const handChest = pose === 'chest'
  const known = pose !== 'unknown'
  return <svg className="lesson-hand-icon" viewBox="0 0 220 190" role="img" aria-label={known ? `Illustration of hand position near ${handPoint ? 'the face' : handMouth ? 'the mouth' : 'the chest'}` : 'Hand position not specified in the source'}>
    <circle className="lesson-head" cx="110" cy="43" r="25" />
    <path className="lesson-body" d="M76 178v-41c0-23 14-37 34-37s34 14 34 37v41M83 112l-23 30m77-30 23 30" />
    {shape === 'both' ? <><path className="lesson-arm" d="M60 142l28-20 15 5m57 15-28-20-15 5"/><circle className="lesson-hand" cx="103" cy="128" r="7"/><circle className="lesson-hand" cx="117" cy="128" r="7"/></>
      : <><path className="lesson-arm" d={handPoint ? 'M160 142l-17-23-13-55' : handMouth ? 'M60 142l27-20 20-54' : handChest ? 'M60 142l27-20 14 17' : 'M60 142l28-20'} />
        <circle className="lesson-hand resting" cx="60" cy="142" r="5" />
        {known && <g className={`lesson-hand ${shape === 'point' ? 'pointing' : 'fingers-close'}`} transform={handPoint ? 'translate(130 52) rotate(270)' : handMouth ? 'translate(106 67) rotate(-5)' : 'translate(100 139)'}>
          {shape === 'point' ? <path d="M-7 8h8V-8c0-5 7-5 7 0V7h11c4 0 5 3 4 6l-4 15c-1 4-4 6-8 6H-3c-5 0-8-3-8-8V13c0-3 1-5 4-5Z"/> : <path d="M-9 8V-5c0-5 7-5 7 0v9-13c0-5 7-5 7 0v13-11c0-5 7-5 7 0V6l2-6c2-4 8-2 6 3l-4 15c-1 5-5 8-10 8H0c-5 0-9-4-9-9Z"/>}
        </g>}
        {!known && <g className="lesson-unknown"><circle cx="110" cy="60" r="17"/><text x="110" y="67" textAnchor="middle">?</text></g>}
      </>}
    {movement === 'down' && <path className="lesson-motion" d="M132 68q9 34 -5 67" markerEnd="url(#lesson-arrow)"/>}
    {movement === 'apart' && <><path className="lesson-motion" d="M105 141 84 154" markerEnd="url(#lesson-arrow)"/><path className="lesson-motion" d="m115 141 21 13" markerEnd="url(#lesson-arrow)"/></>}
    <defs><marker id="lesson-arrow" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto"><path d="M0 0 7 3.5 0 7"/></marker></defs>
    <text className="lesson-position-label" x="110" y="187" textAnchor="middle">{known ? (handPoint ? 'FACE / FOREHEAD' : handMouth ? 'MOUTH / CHEEK' : handChest ? 'CHEST' : 'HAND POSITION') : 'POSITION NOT SPECIFIED'}</text>
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
  const [activeLessonIndex, setActiveLessonIndex] = useState(0)
  const [backendStatus, setBackendStatus] = useState('not configured')
  const backendClassifierReadyRef = useRef(false)

  useEffect(() => {
    if (!API_BASE_URL) return undefined
    const controller = new AbortController()
    fetch(`${API_BASE_URL}/health`, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error('API health check failed')
        return response.json()
      })
      .then((health) => {
        backendClassifierReadyRef.current = Boolean(health.classifier_loaded)
        setBackendStatus(health.classifier_loaded ? 'recognition ready' : 'connected · model weights missing')
      })
      .catch((healthError) => {
        if (healthError.name !== 'AbortError') setBackendStatus('unavailable · using browser model')
      })
    return () => controller.abort()
  }, [])

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
      if (backendClassifierReadyRef.current && API_BASE_URL) {
        const response = await fetch(`${API_BASE_URL}/api/recognize`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            keypoints: samples.map(({ coordinates }) => Array.from({ length: LANDMARK_COUNT }, (_, index) => Array.from(coordinates.slice(index * COORDINATE_COUNT, (index + 1) * COORDINATE_COUNT)))),
          }),
        })
        if (!response.ok) throw new Error(`Backend recognition returned ${response.status}`)
        const result = await response.json()
        const top = result.predictions.map(({ label, confidence }) => ({
          label,
          word: GLOSS_TO_WORD.get(label) || label,
          confidence,
        }))
        setTopPredictions(top)
        if (top[0]?.confidence >= 0.15) {
          const now = Date.now()
          predictionVotesRef.current = [...predictionVotesRef.current, { ...top[0], at: now }].slice(-3)
          const voteCount = predictionVotesRef.current.filter((vote) => vote.label === top[0].label).length
          if (voteCount >= 2) {
            setRecognizedWords((current) => current[0]?.label === top[0].label ? current : [{ ...top[0], at: now }, ...current].slice(0, 8))
          }
        }
        return
      }
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
  const activeLesson = SIGN_LESSONS[activeLessonIndex]

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
            {cameraOn && !hands.length && <HandPlacementGuide />}
            {!cameraOn && <div className="camera-placeholder"><div className="scan-frame"><i className="corner tl" /><i className="corner tr" /><i className="corner bl" /><i className="corner br" /><div className="hand-illustration"><svg viewBox="0 0 120 140" role="img" aria-label="Hand outline"><path d="M34 69V35a7 7 0 0 1 14 0v28-43a7 7 0 0 1 14 0v42-37a7 7 0 0 1 14 0v40-28a7 7 0 0 1 14 0v52c0 27-15 42-39 42h-2c-15 0-26-8-34-23L9 84a8 8 0 0 1 14-8l11 15V69Z" /></svg><span className="orbit orbit-one" /><span className="orbit orbit-two" /><span className="crosshair crosshair-one">+</span><span className="crosshair crosshair-two">+</span></div><div className="placeholder-text"><span className="placeholder-title">Your camera, your canvas</span><span className="placeholder-subtitle">Start the camera to begin tracking</span></div></div></div>}
            {cameraOn && <div className="tracking-badge"><span className="tracking-pulse" /> TRACKING ACTIVE</div>}
            <div className="stage-corner-label">{cameraOn ? `${hands.length} ${hands.length === 1 ? 'HAND' : 'HANDS'} IN FRAME` : 'AWAITING CAMERA'}</div>
          </div>
          <div className="card-controls"><div className="detection-status"><span className={`status-indicator ${cameraOn ? 'on' : ''}`} /><span>{cameraOn ? (hands.length ? `${hands.length} hand${hands.length > 1 ? 's' : ''} detected` : 'Looking for hands…') : 'Camera is off'}</span></div><div className="control-buttons"><button className="icon-button" type="button" onClick={switchCamera} aria-label="Switch camera" title="Switch camera">⇄</button><button className={`camera-button ${cameraOn ? 'stop-button' : ''}`} type="button" onClick={cameraOn ? stopCamera : startCamera} disabled={loading}>{loading ? <><span className="spinner" /> Loading models…</> : cameraOn ? <><span className="stop-icon" /> Stop camera</> : <><span className="camera-icon" /> Start camera</>}</button></div></div>
          {error && <div className="error-message" role="alert">{error}</div>}
          <section className="training-guide" aria-labelledby="training-guide-title">
            <div className="training-guide-heading">
              <span className="guide-book-icon" aria-hidden="true">▤</span>
              <div><span className="dashboard-kicker">FIRST TIME? START HERE</span><h2 id="training-guide-title">How to use a practice session</h2></div>
            </div>
            <ol className="training-steps">
              <li><span>1</span><div><strong>Start the camera</strong><small>Press “Start camera” and allow camera access when your browser asks.</small></div></li>
              <li><span>2</span><div><strong>Set up your hand</strong><small>Face your palm toward the camera. Keep your hand and upper body visible, then make a sign from the word library.</small></div></li>
              <li><span>3</span><div><strong>Read the result</strong><small>Hold a clear view while signing. “Top live match” shows the current guess; “Session words” lists guesses the model recognizes consistently.</small></div></li>
            </ol>
            <p className="training-note"><strong>About practice:</strong> this session checks signs against the existing model. It does not save examples or retrain the model.</p>
          </section>
          <section className="word-dashboard" aria-label="Myanmar sign recognition dashboard">
            <div className="dashboard-heading">
              <div>
                <span className="dashboard-kicker">LIVE RECOGNITION</span>
                <h2>Sign-to-word dashboard</h2>
                <p>Myanmar emergency signs · trained from the local keypoint dataset</p>
              </div>
              <span className={`model-status ${cameraOn ? 'is-ready' : ''}`}><span />{cameraOn ? `${modelStatus || 'Recognition active'}${API_BASE_URL ? ` · API ${backendStatus}` : ''}` : API_BASE_URL ? `API ${backendStatus}` : 'Start tracking to recognize signs'}</span>
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

            <section className="learning-book" aria-label="Myanmar Sign Language learning book">
              <div className="book-heading">
                <div><span className="dashboard-kicker">LEARN A SIGN, STEP BY STEP</span><h3>Myanmar Sign Language learning book</h3><p>Choose a lesson to see the hand shape, position, movement, and facial-expression notes.</p></div>
                <span className="book-count">{SIGN_LESSONS.length} lessons</span>
              </div>
              <div className="book-tabs" role="tablist" aria-label="Sign lessons">
                {SIGN_LESSONS.map((lesson, index) => <button key={lesson.english} id={`lesson-tab-${index}`} type="button" role="tab" aria-selected={activeLessonIndex === index} aria-controls="lesson-page" className={activeLessonIndex === index ? 'is-selected' : ''} onClick={() => setActiveLessonIndex(index)}><span>{String(index + 1).padStart(2, '0')}</span>{lesson.english}</button>)}
              </div>
              <article className="lesson-page" id="lesson-page" role="tabpanel" aria-labelledby={`lesson-tab-${activeLessonIndex}`}>
                <div className="lesson-art"><LessonHandIcon pose={activeLesson.pose} shape={activeLesson.shape} movement={activeLesson.motion} /><span>Hand position illustration</span></div>
                <div className="lesson-content">
                  <div className="lesson-title"><span>LESSON {String(activeLessonIndex + 1).padStart(2, '0')}</span><h4>{activeLesson.english}</h4><strong lang="my">{activeLesson.myanmar}</strong></div>
                  <div className="lesson-facts">
                    <div><span>HAND POSITION</span><p>{activeLesson.pose === 'unknown' ? 'The source does not specify the exact position.' : activeLesson.pose === 'forehead' ? 'Near the eye or forehead; point the index finger toward the face.' : activeLesson.pose === 'eye' ? 'Near the nose or eye area.' : activeLesson.pose === 'mouth' ? 'Near the mouth or cheek, fingers close to the face.' : 'In front of the chest.'}</p></div>
                    <div><span>OTHER HAND</span><p>{activeLesson.other}</p></div>
                    <div><span>MOVEMENT</span><p>{activeLesson.movement}</p></div>
                    <div><span>FACE & PRACTICE</span><p>{activeLesson.face} Practice slowly, watching the full sign and body position.</p></div>
                  </div>
                  <p className="lesson-source">{activeLesson.sourceNote} <a href="https://conferences.info.uaic.ro/consilr/2026/prevEditions/Consilr_2023.pdf" target="_blank" rel="noreferrer">Read the source study ↗</a></p>
                  <p className="lesson-accuracy-note">This hand drawing is a simplified learning illustration. Myanmar Sign Language has regional variants; learn the complete movement from a fluent Deaf signer or teacher.</p>
                </div>
              </article>
            </section>

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

