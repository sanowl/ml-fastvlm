import { type MagnitudeConfig } from 'magnitude-test';
export default {
    url: "http://example.com",
    llm: {
        provider: 'openai-generic',
        options: {
            baseUrl: 'http://localhost:8080',
            apiKey: 'dummy-key-fastvlm-adapter',
            model: 'fastvlm-vision-model',
            temperature: 0.0
        }
    },
    grounding: process.env.MOONDREAM_API_KEY ? {
        provider: 'moondream',
        options: {
            apiKey: process.env.MOONDREAM_API_KEY as string,
            ...(process.env.MOONDREAM_BASE_URL ? { baseUrl: process.env.MOONDREAM_BASE_URL as string } : {})
        }
    } : undefined,

    browser: {
        headless: false,
        slowMo: 0
    }
} satisfies MagnitudeConfig;
