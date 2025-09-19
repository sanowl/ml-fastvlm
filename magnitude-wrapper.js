// Wrapper script to run Magnitude with proper environment variables
const { spawn } = require('child_process');

console.log('Starting Magnitude with FastVLM adapter...');

// Set environment variables
process.env.ANTHROPIC_API_KEY = 'dummy-key-fastvlm-adapter';
process.env.ANTHROPIC_BASE_URL = 'http://localhost:8080';

console.log('Environment variables set:');
console.log('  ANTHROPIC_API_KEY:', process.env.ANTHROPIC_API_KEY);
console.log('  ANTHROPIC_BASE_URL:', process.env.ANTHROPIC_BASE_URL);

// Get command line arguments
const args = process.argv.slice(2);

if (args.length === 0) {
    console.log('Usage: node magnitude-wrapper.js <magnitude-command>');
    console.log('Example: node magnitude-wrapper.js tests/magnitude/simple-test.mag.ts');
    process.exit(1);
}

// Run Magnitude with the environment variables
const magnitude = spawn('npx', ['magnitude', ...args], {
    stdio: 'inherit',
    env: process.env
});

magnitude.on('close', (code) => {
    console.log(`Magnitude exited with code ${code}`);
    process.exit(code);
});

magnitude.on('error', (err) => {
    console.error('Failed to start Magnitude:', err);
    process.exit(1);
});
