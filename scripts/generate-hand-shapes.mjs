import { mkdir, readdir, readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const scriptDir = path.dirname(fileURLToPath(import.meta.url))
const projectDir = path.resolve(scriptDir, '..')
const datasetDir = path.join(projectDir, 'data', 'myanmar-sign-language-recognition')
const outputFile = path.join(projectDir, 'public', 'data', 'msl-hand-shapes.json')
const labels = JSON.parse(await readFile(path.join(datasetDir, 'label_map.json'), 'utf8')).label2idx
const annotationRows = (await readFile(path.join(datasetDir, 'annotations.txt'), 'utf8'))
  .trim()
  .split(/\r?\n/)
  .map((line) => line.split('\t'))

async function collectSequenceFiles(directory) {
  const files = new Map()
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const fullPath = path.join(directory, entry.name)
    if (entry.isDirectory()) {
      for (const [index, file] of await collectSequenceFiles(fullPath)) files.set(index, file)
    } else {
      const match = entry.name.match(/^idx20-(\d+)\.npy$/i)
      if (match) files.set(Number(match[1]), fullPath)
    }
  }
  return files
}

function readNpyShape(buffer) {
  const major = buffer[6]
  const headerLength = major === 1 ? buffer.readUInt16LE(8) : buffer.readUInt32LE(8)
  const headerStart = major === 1 ? 10 : 12
  const header = buffer.subarray(headerStart, headerStart + headerLength).toString('latin1')
  const match = header.match(/'shape':\s*\((\d+),\s*75,\s*3\)/)
  if (!match) throw new Error(`Unexpected keypoint array shape: ${header}`)
  return { frames: Number(match[1]), dataOffset: headerStart + headerLength }
}

function extractHands(buffer) {
  const { frames, dataOffset } = readNpyShape(buffer)
  let bestFrame = -1
  let bestScore = 0
  for (let frame = 0; frame < frames; frame += 1) {
    let score = 0
    for (const start of [33, 54]) {
      const signatures = new Set()
      for (let index = 0; index < 21; index += 1) {
        const offset = dataOffset + ((frame * 75 + start + index) * 3 * 4)
        const point = [0, 1, 2].map((axis) => buffer.readFloatLE(offset + axis * 4))
        signatures.add(point.map((coordinate) => coordinate.toFixed(4)).join(','))
      }
      score += signatures.size
    }
    if (score > bestScore) {
      bestScore = score
      bestFrame = frame
    }
  }
  if (bestFrame < 0) return []

  const hands = []
  for (const [side, start] of [['left', 33], ['right', 54]]) {
    const points = []
    for (let index = 0; index < 21; index += 1) {
      const offset = dataOffset + ((bestFrame * 75 + start + index) * 3 * 4)
      points.push([buffer.readFloatLE(offset), buffer.readFloatLE(offset + 4)])
    }
    const uniquePoints = new Set(points.map((point) => point.map((coordinate) => coordinate.toFixed(4)).join(',')))
    if (uniquePoints.size >= 10) {
      hands.push({ side, points })
    }
  }
  return hands
}

const sequenceFiles = await collectSequenceFiles(path.join(datasetDir, 'keypoints'))
const shapes = {}
for (const [id, file] of sequenceFiles) {
  const row = annotationRows[id - 1]
  const gloss = row?.[1]?.trim()
  if (!gloss || !(gloss in labels) || shapes[gloss]) continue
  const hands = extractHands(await readFile(file))
  if (hands.length) {
    shapes[gloss] = hands.map(({ points }) => points.map(([x, y]) => [
      Math.round(x * 1000) / 1000,
      Math.round(y * 1000) / 1000,
    ]))
  }
}

await mkdir(path.dirname(outputFile), { recursive: true })
await writeFile(outputFile, `${JSON.stringify(shapes)}\n`, 'utf8')
console.log(`Wrote ${Object.keys(shapes).length} sign hand shapes to ${path.relative(projectDir, outputFile)}`)
