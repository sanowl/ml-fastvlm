try { require('dotenv').config(); } catch {}

const { startBrowserAgent } = require('magnitude-core');

function parseArgs() {
  const args = process.argv.slice(2);
  const out = { url: '', task: '', headless: false };
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === '--url') out.url = args[++i];
    else if (a === '--task') out.task = args[++i];
    else if (a === '--headless') out.headless = true;
  }
  if (!out.url || !out.task) {
    console.log('Usage: node scripts/ask.js --url <start_url> --task "<what to do>" [--headless]');
    process.exit(1);
  }
  return out;
}

async function main() {
  const { url, task, headless } = parseArgs();

  const baseUrl = process.env.ANTHROPIC_BASE_URL || 'http://localhost:8080';
  const apiKey = process.env.ANTHROPIC_API_KEY || 'dummy-key-fastvlm-adapter';

  console.log('Starting BrowserAgent with FastVLM adapter...');
  // Diagnostics: show effective env (without leaking secrets)
  const redact = (v) => (typeof v === 'string' && v.length > 12) ? `${v.slice(0,6)}…${v.slice(-4)}` : (v || '');
  console.log('Env check:');
  console.log('  ANTHROPIC_BASE_URL =', baseUrl);
  console.log('  FASTVLM_DISABLE_NAV_OVERRIDE =', process.env.FASTVLM_DISABLE_NAV_OVERRIDE ? '1' : '0');
  console.log('  MOONDREAM_API_KEY =', process.env.MOONDREAM_API_KEY ? `(set ${redact(process.env.MOONDREAM_API_KEY)})` : '(not set)');
  if (process.env.MOONDREAM_BASE_URL) {
    console.log('  MOONDREAM_BASE_URL =', process.env.MOONDREAM_BASE_URL);
  }
  const opts = {
    // LLM points at our adapter (OpenAI-compatible shape)
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
    // Browser settings
    headless,
    enableVision: true,
  };

  // If MOONDREAM_API_KEY is set, enable grounding for click-based actions
  if (process.env.MOONDREAM_API_KEY) {
    console.log('Enabling grounding (Moondream) for element clicks.');
    const groundingOptions = { apiKey: process.env.MOONDREAM_API_KEY };
    if (process.env.MOONDREAM_BASE_URL) groundingOptions.baseUrl = process.env.MOONDREAM_BASE_URL;
    opts.grounding = { provider: 'moondream', options: groundingOptions };
  } else {
    console.log('No MOONDREAM_API_KEY set — using nav/keyboard fallback (less reliable for complex UIs).');
  }

  // Improve screenshot stability for grounding
  opts.virtualScreenDimensions = { width: 1280, height: 800 };

  const agent = await startBrowserAgent(opts);

  try {
    console.log('Navigating to:', url);
    await agent.nav(url);

    console.log('Acting:', task);
    try {
      await agent.act(task);
    } catch (err) {
      console.error('Action error message:', err?.message || err);
      if (err && err.stack) console.error('Action error stack:', err.stack);
      if (err && err.response) {
        try {
          console.error('Action error response status:', err.response.status);
          console.error('Action error response headers:', JSON.stringify(err.response.headers));
          if (err.response.data) console.error('Action error response data:', JSON.stringify(err.response.data));
          if (err.response.text) console.error('Action error response text:', err.response.text);
        } catch {}
      }
      throw err;
    }

    console.log('Done. You can close the browser now.');
  } catch (err) {
    console.error('Error running task:', err?.message || err);
  } finally {
    try { await agent.stop(); } catch {}
  }
}

main();
