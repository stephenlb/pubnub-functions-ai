// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
// Imports
// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
const ai = require('./ai.js');

// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
// Main
// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

function main() {
    const training = generateTestTrainingSampleSet(2000);
    const testing  = generateTestTrainingSampleSet(16);

    // AI Training
    const nn = new ai.NeuralNet({ type : 'tanh', learn : 0.001 });
    nn.train({ dataset: training, epochs: 10000, batchSize: 10 });

    // need a vectorizor
    // makes sure time +1 so minute/hour/day don't zero out

    // Predict
    let inputs      = testing.map( m => m[0] );
    let targets     = testing.map( m => m[1] );
    let predictions = nn.predict(inputs).map(a => a.map(a => Math.round(a)));

    console.log('\n');
    console.log('predictions', predictions.map(m => m[0]));
    console.log('targets    ', targets.map(m => m[0]));
    console.log('\n');
    console.log('size', nn.matrixSize());
    console.log('\n');
}

// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
// Generate Test Data
// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
function generateTestTrainingSampleSet(samples) {
    return ai.Tensor(samples, i => generateTestTrainingSample() );
}
    
function generateTestTrainingSample() {
    // Ranges of time
    const days    =  7; // Day of Week 1-7
    const hours   = 24; // Hour
    const minutes = 60; // Minute

    // Bit depths
    const dayBits    = 7; // Day of Week 1-7 OneHot
    const hourBits   = 5; // Hour            Encoded
    const minuteBits = 6; // Minute          Encoded

    // Random Sample Generation
    const day    = rnd(days);
    const hour   = rnd(hours);
    const minute = rnd(minutes);

    // Create Vectorized Sample
    const input = []
        .concat(ai.Tensor(dayBits,    i => day === 1 + i           ? 1 : 0))
        .concat(ai.Tensor(hourBits,   i => Math.pow(2, i) & hour   ? 1 : 0))
        .concat(ai.Tensor(minuteBits, i => Math.pow(2, i) & minute ? 1 : 0));
    const base    = input[0] || input[1] ? 80 : 40;
    const output  = [base - rnd(2) + rnd(2)];

    return [input, output];
}

// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
// Simple Random Function
// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
function rnd(r) {
    return Math.ceil(Math.random() * r);
}

// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
// Run Main
// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
main();
