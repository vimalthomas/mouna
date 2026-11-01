import { Injectable } from '@angular/core';

export interface AslGuide {
  word: string;
  instruction: string;
  handMarkers: HandMarker[];
  arrows: Arrow[];
}

export interface HandMarker {
  x: number;
  y: number;
  label: 'L' | 'R' | 'LR';
  shape: 'open' | 'fist' | 'point' | 'flat' | 'c' | 'y' | 'v';
}

export interface Arrow {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  style: 'straight' | 'arc' | 'wave';
}

// Body reference points (in 0-100 coordinate space)
// Head center: 50,15  Forehead: 50,10  Chin: 50,25  Mouth: 50,22
// Chest: 50,45  Stomach: 50,60  Shoulders: 30,35 / 70,35
// Ears: 35,15 / 65,15  Cheeks: 40,20 / 60,20  Temple: 38,12 / 62,12
// Front space: 50,50  Side: 25,50 / 75,50

const guides: Record<string, AslGuide> = {
  hello: {
    word: 'hello',
    instruction: 'Open hand at forehead, wave outward',
    handMarkers: [{ x: 60, y: 12, label: 'R', shape: 'open' }],
    arrows: [{ x1: 60, y1: 12, x2: 78, y2: 15, style: 'wave' }],
  },
  goodbye: {
    word: 'goodbye',
    instruction: 'Open hand in front, wave fingers open/close',
    handMarkers: [{ x: 55, y: 40, label: 'R', shape: 'open' }],
    arrows: [{ x1: 55, y1: 38, x2: 55, y2: 42, style: 'wave' }],
  },
  thank_you: {
    word: 'thank_you',
    instruction: 'Flat hand touches chin, moves forward and down',
    handMarkers: [{ x: 50, y: 25, label: 'R', shape: 'flat' }],
    arrows: [{ x1: 50, y1: 25, x2: 60, y2: 40, style: 'straight' }],
  },
  please: {
    word: 'please',
    instruction: 'Flat hand on chest, circular motion',
    handMarkers: [{ x: 50, y: 45, label: 'R', shape: 'flat' }],
    arrows: [{ x1: 45, y1: 42, x2: 55, y2: 48, style: 'arc' }],
  },
  sorry: {
    word: 'sorry',
    instruction: 'Fist on chest, circular motion',
    handMarkers: [{ x: 50, y: 45, label: 'R', shape: 'fist' }],
    arrows: [{ x1: 45, y1: 42, x2: 55, y2: 48, style: 'arc' }],
  },
  yes: {
    word: 'yes',
    instruction: 'Fist nods up and down like a head',
    handMarkers: [{ x: 55, y: 40, label: 'R', shape: 'fist' }],
    arrows: [{ x1: 55, y1: 35, x2: 55, y2: 48, style: 'wave' }],
  },
  no: {
    word: 'no',
    instruction: 'Index + middle finger snap to thumb',
    handMarkers: [{ x: 55, y: 38, label: 'R', shape: 'point' }],
    arrows: [{ x1: 58, y1: 36, x2: 52, y2: 40, style: 'straight' }],
  },
  help: {
    word: 'help',
    instruction: 'Fist on open palm, lift both hands up',
    handMarkers: [
      { x: 48, y: 50, label: 'L', shape: 'flat' },
      { x: 48, y: 46, label: 'R', shape: 'fist' },
    ],
    arrows: [{ x1: 48, y1: 50, x2: 48, y2: 35, style: 'straight' }],
  },
  want: {
    word: 'want',
    instruction: 'Both hands out, fingers spread, pull toward body',
    handMarkers: [
      { x: 38, y: 45, label: 'L', shape: 'open' },
      { x: 62, y: 45, label: 'R', shape: 'open' },
    ],
    arrows: [
      { x1: 38, y1: 45, x2: 42, y2: 50, style: 'straight' },
      { x1: 62, y1: 45, x2: 58, y2: 50, style: 'straight' },
    ],
  },
  like: {
    word: 'like',
    instruction: 'Thumb + middle finger on chest, pull away closing',
    handMarkers: [{ x: 50, y: 45, label: 'R', shape: 'open' }],
    arrows: [{ x1: 50, y1: 45, x2: 62, y2: 42, style: 'straight' }],
  },
  love: {
    word: 'love',
    instruction: 'Cross both arms over chest, hug yourself',
    handMarkers: [
      { x: 60, y: 45, label: 'L', shape: 'fist' },
      { x: 40, y: 45, label: 'R', shape: 'fist' },
    ],
    arrows: [
      { x1: 35, y1: 40, x2: 60, y2: 45, style: 'arc' },
      { x1: 65, y1: 40, x2: 40, y2: 45, style: 'arc' },
    ],
  },
  friend: {
    word: 'friend',
    instruction: 'Hook index fingers together, flip positions',
    handMarkers: [
      { x: 45, y: 45, label: 'L', shape: 'point' },
      { x: 55, y: 45, label: 'R', shape: 'point' },
    ],
    arrows: [{ x1: 45, y1: 43, x2: 55, y2: 47, style: 'arc' }],
  },
  mother: {
    word: 'mother',
    instruction: 'Open hand, tap thumb on chin',
    handMarkers: [{ x: 50, y: 25, label: 'R', shape: 'open' }],
    arrows: [{ x1: 50, y1: 23, x2: 50, y2: 27, style: 'wave' }],
  },
  father: {
    word: 'father',
    instruction: 'Open hand, tap thumb on forehead',
    handMarkers: [{ x: 50, y: 12, label: 'R', shape: 'open' }],
    arrows: [{ x1: 50, y1: 10, x2: 50, y2: 14, style: 'wave' }],
  },
  baby: {
    word: 'baby',
    instruction: 'Rock arms as if holding a baby',
    handMarkers: [
      { x: 40, y: 50, label: 'L', shape: 'flat' },
      { x: 60, y: 48, label: 'R', shape: 'flat' },
    ],
    arrows: [{ x1: 45, y1: 48, x2: 55, y2: 52, style: 'arc' }],
  },
  eat: {
    word: 'eat',
    instruction: 'Pinched fingers tap mouth repeatedly',
    handMarkers: [{ x: 50, y: 22, label: 'R', shape: 'flat' }],
    arrows: [{ x1: 55, y1: 30, x2: 50, y2: 22, style: 'wave' }],
  },
  drink: {
    word: 'drink',
    instruction: 'C-hand tilts to mouth like holding a cup',
    handMarkers: [{ x: 52, y: 22, label: 'R', shape: 'c' }],
    arrows: [{ x1: 55, y1: 30, x2: 52, y2: 22, style: 'arc' }],
  },
  water: {
    word: 'water',
    instruction: 'W-hand, index finger taps chin',
    handMarkers: [{ x: 50, y: 25, label: 'R', shape: 'open' }],
    arrows: [{ x1: 50, y1: 23, x2: 50, y2: 27, style: 'wave' }],
  },
  book: {
    word: 'book',
    instruction: 'Palms together, open outward like a book',
    handMarkers: [
      { x: 45, y: 45, label: 'L', shape: 'flat' },
      { x: 55, y: 45, label: 'R', shape: 'flat' },
    ],
    arrows: [
      { x1: 45, y1: 45, x2: 35, y2: 40, style: 'straight' },
      { x1: 55, y1: 45, x2: 65, y2: 40, style: 'straight' },
    ],
  },
  dog: {
    word: 'dog',
    instruction: 'Snap fingers and pat thigh',
    handMarkers: [{ x: 55, y: 50, label: 'R', shape: 'open' }],
    arrows: [{ x1: 55, y1: 45, x2: 60, y2: 65, style: 'straight' }],
  },
  cat: {
    word: 'cat',
    instruction: 'Pinch at cheek, pull outward like whiskers',
    handMarkers: [{ x: 58, y: 20, label: 'R', shape: 'point' }],
    arrows: [{ x1: 58, y1: 20, x2: 72, y2: 20, style: 'straight' }],
  },
  happy: {
    word: 'happy',
    instruction: 'Flat hand brushes up on chest repeatedly',
    handMarkers: [{ x: 50, y: 45, label: 'R', shape: 'flat' }],
    arrows: [{ x1: 50, y1: 50, x2: 50, y2: 38, style: 'wave' }],
  },
  sad: {
    word: 'sad',
    instruction: 'Both open hands drag down in front of face',
    handMarkers: [
      { x: 42, y: 18, label: 'L', shape: 'open' },
      { x: 58, y: 18, label: 'R', shape: 'open' },
    ],
    arrows: [
      { x1: 42, y1: 18, x2: 42, y2: 32, style: 'straight' },
      { x1: 58, y1: 18, x2: 58, y2: 32, style: 'straight' },
    ],
  },
  good: {
    word: 'good',
    instruction: 'Flat hand touches chin, moves down to open palm',
    handMarkers: [
      { x: 45, y: 50, label: 'L', shape: 'flat' },
      { x: 50, y: 25, label: 'R', shape: 'flat' },
    ],
    arrows: [{ x1: 50, y1: 25, x2: 45, y2: 50, style: 'straight' }],
  },
  bad: {
    word: 'bad',
    instruction: 'Flat hand touches chin, flips hand over downward',
    handMarkers: [{ x: 50, y: 25, label: 'R', shape: 'flat' }],
    arrows: [{ x1: 50, y1: 25, x2: 50, y2: 42, style: 'straight' }],
  },
  phone: {
    word: 'phone',
    instruction: 'Y-hand held to ear like a phone',
    handMarkers: [{ x: 63, y: 15, label: 'R', shape: 'y' }],
    arrows: [],
  },
  home: {
    word: 'home',
    instruction: 'Pinched hand touches near mouth, then near ear',
    handMarkers: [{ x: 52, y: 22, label: 'R', shape: 'flat' }],
    arrows: [{ x1: 52, y1: 22, x2: 62, y2: 16, style: 'straight' }],
  },
  school: {
    word: 'school',
    instruction: 'Clap hands together twice',
    handMarkers: [
      { x: 47, y: 45, label: 'L', shape: 'flat' },
      { x: 53, y: 45, label: 'R', shape: 'flat' },
    ],
    arrows: [{ x1: 47, y1: 43, x2: 53, y2: 47, style: 'wave' }],
  },
  work: {
    word: 'work',
    instruction: 'Dominant fist taps on top of other fist',
    handMarkers: [
      { x: 50, y: 50, label: 'L', shape: 'fist' },
      { x: 50, y: 44, label: 'R', shape: 'fist' },
    ],
    arrows: [{ x1: 50, y1: 42, x2: 50, y2: 48, style: 'wave' }],
  },
  stop: {
    word: 'stop',
    instruction: 'Flat blade hand chops down on open palm',
    handMarkers: [
      { x: 48, y: 48, label: 'L', shape: 'flat' },
      { x: 52, y: 40, label: 'R', shape: 'flat' },
    ],
    arrows: [{ x1: 52, y1: 36, x2: 48, y2: 48, style: 'straight' }],
  },
  go: {
    word: 'go',
    instruction: 'Both index fingers point and arc forward',
    handMarkers: [
      { x: 42, y: 45, label: 'L', shape: 'point' },
      { x: 58, y: 45, label: 'R', shape: 'point' },
    ],
    arrows: [
      { x1: 42, y1: 45, x2: 35, y2: 35, style: 'arc' },
      { x1: 58, y1: 45, x2: 65, y2: 35, style: 'arc' },
    ],
  },
  walk: {
    word: 'walk',
    instruction: 'Flat hands alternate like feet walking',
    handMarkers: [
      { x: 45, y: 55, label: 'L', shape: 'flat' },
      { x: 55, y: 52, label: 'R', shape: 'flat' },
    ],
    arrows: [
      { x1: 45, y1: 55, x2: 45, y2: 48, style: 'wave' },
      { x1: 55, y1: 52, x2: 55, y2: 58, style: 'wave' },
    ],
  },
  think: {
    word: 'think',
    instruction: 'Point index finger at temple, small circle',
    handMarkers: [{ x: 62, y: 13, label: 'R', shape: 'point' }],
    arrows: [{ x1: 60, y1: 11, x2: 64, y2: 15, style: 'arc' }],
  },
  know: {
    word: 'know',
    instruction: 'Flat fingertips tap forehead',
    handMarkers: [{ x: 52, y: 10, label: 'R', shape: 'flat' }],
    arrows: [{ x1: 52, y1: 8, x2: 52, y2: 13, style: 'wave' }],
  },
  see: {
    word: 'see',
    instruction: 'V-hand points from eyes outward',
    handMarkers: [{ x: 55, y: 16, label: 'R', shape: 'v' }],
    arrows: [{ x1: 55, y1: 16, x2: 68, y2: 20, style: 'straight' }],
  },
  learn: {
    word: 'learn',
    instruction: 'Grab from palm, bring up to forehead',
    handMarkers: [
      { x: 45, y: 48, label: 'L', shape: 'flat' },
      { x: 48, y: 48, label: 'R', shape: 'flat' },
    ],
    arrows: [{ x1: 48, y1: 48, x2: 50, y2: 12, style: 'straight' }],
  },
  sleep: {
    word: 'sleep',
    instruction: 'Open hand slides down face, closes at chin',
    handMarkers: [{ x: 50, y: 16, label: 'R', shape: 'open' }],
    arrows: [{ x1: 50, y1: 16, x2: 50, y2: 28, style: 'straight' }],
  },
  red: {
    word: 'red',
    instruction: 'Index finger strokes down from lips',
    handMarkers: [{ x: 50, y: 22, label: 'R', shape: 'point' }],
    arrows: [{ x1: 50, y1: 22, x2: 50, y2: 30, style: 'straight' }],
  },
  blue: {
    word: 'blue',
    instruction: 'B-hand twists side to side in open space',
    handMarkers: [{ x: 60, y: 40, label: 'R', shape: 'flat' }],
    arrows: [{ x1: 55, y1: 40, x2: 65, y2: 40, style: 'wave' }],
  },
  yellow: {
    word: 'yellow',
    instruction: 'Y-hand twists side to side',
    handMarkers: [{ x: 60, y: 40, label: 'R', shape: 'y' }],
    arrows: [{ x1: 55, y1: 40, x2: 65, y2: 40, style: 'wave' }],
  },
};

@Injectable({ providedIn: 'root' })
export class AslGuideService {
  getGuide(word: string): AslGuide | null {
    const key = word.toLowerCase().replace(/ /g, '_');
    return guides[key] || null;
  }

  hasGuide(word: string): boolean {
    return word.toLowerCase().replace(/ /g, '_') in guides;
  }
}
