import { useCallback, useEffect, useRef, useState } from 'react'
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision'
import './App.css'

const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
]

export default function App() {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const landmarkerRef = useRef(null)
  const frameRef = useRef(0)
  const facingModeRef = useRef('user')
  const [cameraOn, setCameraOn] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [hands, setHands] = useState([])
  const [fps, setFps] = useState(0)
  const [facingMode, setFacingMode] = useState('user')

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
  }, [])

  const startCamera = useCallback(async () => {
    setError('')
    setLoading(true)
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera access requires a secure connection (HTTPS or localhost).')
      if (!landmarkerRef.current) {
        const vision = await FilesetResolver.forVisionTasks('/mediapipe/wasm')
        landmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
          runningMode: 'VIDEO',
          numHands: 2,
          minHandDetectionConfidence: 0.55,
          minHandPresenceConfidence: 0.55,
          minTrackingConfidence: 0.5,
        })
      }
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
  }, [facingMode, stopCamera])

  useEffect(() => {
    if (!cameraOn) return undefined
    let lastTime = 0
    let fpsCount = 0
    let fpsStart = performance.now()
    const render = () => {
      const video = videoRef.current
      const canvas = canvasRef.current
      const detector = landmarkerRef.current
      if (!video || !canvas || !detector || video.readyState < 2) {
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
        const detectedHands = result.landmarks || []
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
  }, [cameraOn])

  useEffect(() => () => {
    if (videoRef.current?.srcObject) videoRef.current.srcObject.getTracks().forEach((track) => track.stop())
    landmarkerRef.current?.close()
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
          <p className="hero-description">See hand tracking in action. Move your hands into frame and watch 21 key landmarks map every movement, instantly.</p>
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
          <div className="card-controls"><div className="detection-status"><span className={`status-indicator ${cameraOn ? 'on' : ''}`} /><span>{cameraOn ? (hands.length ? `${hands.length} hand${hands.length > 1 ? 's' : ''} detected` : 'Looking for hands…') : 'Camera is off'}</span></div><div className="control-buttons"><button className="icon-button" type="button" onClick={switchCamera} aria-label="Switch camera" title="Switch camera">⇄</button><button className={`camera-button ${cameraOn ? 'stop-button' : ''}`} type="button" onClick={cameraOn ? stopCamera : startCamera} disabled={loading}>{loading ? <><span className="spinner" /> Loading model…</> : cameraOn ? <><span className="stop-icon" /> Stop camera</> : <><span className="camera-icon" /> Start camera</>}</button></div></div>
          {error && <div className="error-message" role="alert">{error}</div>}
        </div>
      </section>

      <section className="details-row" aria-label="Tracking details"><div className="detail-item"><span className="detail-number">01</span><div><span className="detail-title">21 LANDMARKS</span><span className="detail-copy">Every joint, mapped</span></div></div><span className="detail-separator" /><div className="detail-item"><span className="detail-number">02</span><div><span className="detail-title">2 HANDS</span><span className="detail-copy">Tracked simultaneously</span></div></div><span className="detail-separator" /><div className="detail-item"><span className="detail-number">03</span><div><span className="detail-title">ON-DEVICE</span><span className="detail-copy">Private by design</span></div></div><div className="powered-by"><span className="powered-line" /> POWERED BY <strong>MEDIAPIPE</strong></div></section>
      <footer className="footer"><span>MADE FOR A MORE CONNECTED WORLD</span><span>HAND LANDMARKER <b>·</b> WEB DEMO</span></footer>
    </main>
  )
}
