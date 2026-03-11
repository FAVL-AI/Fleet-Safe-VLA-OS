import fs from 'fs';
import { DOMParser } from 'xmldom';
import URDFLoader from 'urdf-loader';

const loader = new URDFLoader();
loader.parseOptions = { parseDOM: (content) => new DOMParser().parseFromString(content, 'text/xml') };

try {
  const content = fs.readFileSync('public/robots/unitree_g1/g1_29dof.urdf', 'utf8');
  const robot = loader.parse(content);
  console.log('Successfully parsed URDF. Object keys:', Object.keys(robot));
} catch (e) {
  console.error('Failed to parse URDF:', e);
}
