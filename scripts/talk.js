try { require('dotenv').config(); } catch {}

const readline = require('readline');
const { startBrowserAgent } = require('magnitude-core');

function parseArgs() {
  const args = process.argv.slice(2);
  const out = { url: '', headless: false };
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === '--url') out.url = args[++i];
    else if (a === '--headless') out.headless = true;
  }
  return out;
}

async function main() {
  const { url, headless } = parseArgs();
  const baseUrl = process.env.ANTHROPIC_BASE_URL || 'http://localhost:8080';
  const apiKey = process.env.ANTHROPIC_API_KEY || 'dummy-key-fastvlm-adapter';

  console.log('Starting BrowserAgent (interactive) with FastVLM adapter...');
  // Diagnostics: show effective env (without leaking secrets)
  const redact = (v) => (typeof v === 'string' && v.length > 12) ? `${v.slice(0,6)}…${v.slice(-4)}` : (v || '');
  console.log('Env check:');
  console.log('  ANTHROPIC_BASE_URL =', process.env.ANTHROPIC_BASE_URL || '(default http://localhost:8080)');
  console.log('  FASTVLM_DISABLE_NAV_OVERRIDE =', process.env.FASTVLM_DISABLE_NAV_OVERRIDE ? '1' : '0');
  console.log('  MOONDREAM_API_KEY =', process.env.MOONDREAM_API_KEY ? `(set ${redact(process.env.MOONDREAM_API_KEY)})` : '(not set)');
  if (process.env.MOONDREAM_BASE_URL) {
    console.log('  MOONDREAM_BASE_URL =', process.env.MOONDREAM_BASE_URL);
  }
  const opts = {
    llm: {
      provider: 'openai-generic',
      options: {
        baseUrl,
        apiKey,
        model: 'fastvlm-vision-model',
        temperature: 0.0,
        supportsVision: true,
      },
    },
    headless,
    enableVision: true,
  };
  if (process.env.MOONDREAM_API_KEY) {
    console.log('Enabling grounding (Moondream) for element clicks.');
    const groundingOptions = { apiKey: process.env.MOONDREAM_API_KEY };
    if (process.env.MOONDREAM_BASE_URL) groundingOptions.baseUrl = process.env.MOONDREAM_BASE_URL;
    opts.grounding = { provider: 'moondream', options: groundingOptions };
  } else {
    console.log('No MOONDREAM_API_KEY set — using nav/keyboard fallback (less reliable for complex UIs).');
  }
  
  opts.virtualScreenDimensions = { width: 1280, height: 800 };

  const agent = await startBrowserAgent(opts);
  try {
    if (url) {
      console.log('Navigating to:', url);
      await agent.nav(url);
    }

    console.log("Type instructions (or 'exit' to quit):");
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout, prompt: '> ' });
    rl.prompt();
    rl.on('line', async (line) => {
      const cmd = line.trim();
      if (!cmd) return rl.prompt();
      if (cmd.toLowerCase() === 'exit' || cmd.toLowerCase() === 'quit') {
        rl.close();
        return;
      }
      try {
        console.log('Acting:', cmd);
        await agent.act(cmd);
      } catch (err) {

        // Expanded diagnostics for 400 errors (often from grounding provider)
        console.error('Action error message:', err?.message || err);
        if (err && err.stack) console.error('Action error stack:', err.stack);
        if (err && err.response) {
          try {
            console.error('Action error response status:', err.response.status);
            console.error('Action error response headers:', JSON.stringify(err.response.headers));
            // Some libs put data/text on response
            if (err.response.data) console.error('Action error response data:', JSON.stringify(err.response.data));
            if (err.response.text) console.error('Action error response text:', err.response.text);
          } catch {}
        }

        // If grounding fails, try some common fallbacks
        if (err?.message?.includes('Moondream returned no points') ||
            err?.message?.includes('target unclear')) {
          console.log('🔄 Grounding failed, trying fallback strategies...');

          try {
            // Try some common alternatives
            const fallbacks = [
              'Continue',
              'Search',
              'Accept',
              'OK'
            ];

            for (const fallback of fallbacks) {
              try {
                console.log(` Trying fallback: "${fallback}"`);
                await agent.act(`Click "${fallback}"`);
                console.log(` Fallback "${fallback}" succeeded`);
                break;
              } catch (fallbackErr) {
                console.log(`Fallback "${fallback}" failed`);
                continue;
              }
            }
          } catch (fallbackError) {
            console.log('All fallbacks failed');
          }
        }
      }
      rl.prompt();
    });
    await new Promise((resolve) => rl.on('close', resolve));
  } finally {
    try { await agent.stop(); } catch {}
  }
}

main().catch((err) => {
  console.error('Fatal:', err?.stack || err?.message || err);
  process.exit(1);
});
