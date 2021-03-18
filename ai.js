// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
// Visualizer
// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class Visualizer {
    /**
     * Initialize visualizer.
     *
     * @param {Object} options
     * @return {Object} this
     */
    constructor (options) {}

    /**
     * Visualize Training Data
     * @param {Array} data
     */
    static training (data) {
        Tensor(20, () => 0).map(i => console.log(
            JSON.stringify(data[Math.floor(Math.random()*data.length)])
                .replace(/\[1\]/g,'✅')
                .replace(/\[0\]/g,'🛑')
                .replace(/[\[\]', ]/g,'')
                .replace(/0/g,'▗')
                .replace(/1/g,'▇')
        ));
    }

    /**
     * Visualize learning loss results after training
     * @param {Object} neuralNet instance
     */
    static loss (neuralNet) {
        let nn         = neuralNet;
        let asciichart = require('asciichart');
        let mod        = Math.round(nn.losses.length/90);
        let loss       = nn.losses.filter((l,i)=>!(i%mod)).map(l=>100*(1-l));

        console.log(asciichart.plot(loss, {height : 20}));
        console.log('\n');
    }
}

// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
// Profiler
// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class Profiler {
    /**
     * Initialize profiler.
     *
     * @param {String} name of the profiler
     * @return {Object} this
     */
    constructor (name) {
        this.name  = setup.name || "Unamed Profile";
        this.start = +new Date();
    }

    analyze () {
        console.log('⏱  '+this.name+': ', (+new Date() - this.start)+'ms');
    }
}

// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
// Importer - imports common file types like csv and tensorflow
// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class Importer {
    /**
     * Load File Data.
     *
     * @param {String} name of the file
     * @return {Promise} promise with resulting file data
     */
    static loadFile (fileName) {
        let fs = require('fs');
        return new Promise((resolve, reject) => {
            fs.readFile( __dirname + '/' + fileName, (err, data) => {
                if (err) reject(err); 
                else     resolve(data.toString());
            });
        });
    }

    /**
     * Load and Decode a JSON File.
     *
     * @param {String} name of the file
     * @return {Promise} promise with resulting file data
     */
    static loadJSON (fileName) {
        return new Promise((resolve, reject) => {
            self.loadFile(fileName)
                .catch(err => reject(err))
                .then(data => {
                    try      {resolve(JSON.parse(data));}
                    catch(e) {reject(e);}
                });
        });
    }
}

// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
// Vectorizer methods to encode/decode for input/output in a NeuralNet
// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class Vectorizer {
    /**
     * Initialize vectorizer.
     *
     * @return {Object} this
     */
    constructor () {
        // Basic Alphabet
        this.ctrx       = /[^a-z@. ]+/g;
        this.cletter    = 'abcdefghijklmnopqrstuvwxyz ';
        this.consonants = 'bcdfghjklmnpqrstvwxz';
        this.vowels     = 'aeiouy';
        this.clmap      = {};
        this.comap      = {};
        this.vomap      = {};

        // Map Letters
        this.cletter.split('')   .forEach((l, i) => this.clmap[l]=i);
        this.consonants.split('').forEach((l, i) => this.comap[l]=i);
        this.vowels.split('')    .forEach((l, i) => this.vomap[l]=i);

        // Basic Alphanumeric
        this.alphanemeric = 'abcdefghijklmnopqrstuvwxyz0123456789@.-_ ';
        this.anrx         = /[^a-z0-9@.\-_ ]+/g;
        this.anmap        = {};
        this.alphanemeric.split('').forEach((l, i) => this.anmap[l]=i);

        // Sentence Features Array
        this.ssf = [/[a-z0-9]\.[a-z]/, /[a-z0-9]@[a-z0-9]/, / ?email:? ?/];
    }
    
    /**
     * Compact Tokenizer for Long Sentences.
     * Faster learning speeds.
     * Less overfitting for false negatives.
     *
     * @param {Object} setup parameters
     * @return {Tensor} a one-hot array of vectorized inputs
     */
    charType (setup) {
        let alen     = 4;
        let collapse = setup.collapse  || 5;
        let maxLen   = setup.maxLength || 400;
        let size     = setup.size      || Math.floor((maxLen / collapse) * alen);
        let matrix   = Tensor(size, i => 0);
        let sentence = setup.sentence
                            .toLowerCase()
                            .replace(this.ctrx, '')
                            .split('');

        sentence.slice(0, maxLen).forEach((l, i) => {
            let group = Math.floor(i / collapse) * alen;
            if      (this.clmap[l]) matrix[group]   = 1;
            else if (l==='@')       matrix[group+1] = 1;
            else if (l===' ')       matrix[group+2] = 1;
            else if (l==='.')       matrix[group+3] = 1;
        });

        return matrix;
    };

    /**
     * Alpha Position Big One-hot.
     *
     * @param {String} string of letters
     * @return {Tensor} a one-hot array of vectorized inputs
     */
    alpha (string) {
        let len = this.cletter.length;
        return [].concat.apply( [], string.split('').map( letter =>
            Tensor(len, i => this.clmap[letter] === i ? 1 : 0)
        ) );
    }

    /**
     * Sentence Feature and Structures.
     *
     * @param {Object} setup parameters
     * @return {Tensor} a one-hot array of vectorized inputs
     */
    sentenceStruct (setup) {
        return this.ssf.map((rx, i) =>
            setup.sentence.toLowerCase().match(rx)?1:0);
    }

    // 
    /**
     * Big Sentence Vectorizer ( may have too much overfitting ).
     *
     * @param {Object} setup parameters
     * @return {Tensor} a one-hot array of vectorized inputs
     */
    sentence (setup) {
        let sentence = setup.sentence.toLowerCase().replace(anrx, '').split('');
        let alen     = alphanemeric.length;
        let collapse = setup.collapse  || 20;
        let maxLen   = setup.maxLength || 400;
        let size     = setup.size      || Math.floor((maxLen / collapse) * alen);
        let matrix   = Tensor(size, i => 0);

        // 40 floats with size 100 representing 10 chars each 40 floats
        sentence.slice(0, maxLen).forEach((l, i) => {
            let group = Math.floor(i / collapse);
            matrix[this.anmap[l]+(alen*group)] = 1;
        });

        return matrix;
    };

    /**
     * Big Number Vectorizer.
     *
     * @param {Number} Number
     * @return {Tensor} a one-hot array of vectorized inputs
     */
    number (number) {
        return Tensor(10, i => number >> i & 1);
    }

    /**
     * Consonants and Vowels Vectorization.
     *
     * @param {String} Sentence
     * @return {Tensor} a one-hot array of vectorized letter sequence
     */
    cv (string) {
        return [].concat.apply([], string.split('').map( letter => [
            +(letter in this.comap)
        ,   +(letter in this.vomap)
        ]));
    }

    /**
     * Consonants and Vowels Vectorization with High/Low bits.
     *
     * @param {String} Sentence
     * @return {Tensor} a one-hot array of vectorized letter sequence
     */
    cvhigh (string) {
        let colen = this.consonants.length;
        let volen = this.vowels.length;
        return [].concat.apply([], string.split('').map( letter => [
            +(letter in this.comap)
        ,   +(letter in this.vomap)
        ,   +(letter in this.comap && this.comap[letter] >= (colen/2))
        ,   +(letter in this.vomap && this.vomap[letter] >= (volen/2))
        ]));
    }

    /**
     * Five-bit Sentence binary encoding.
     *
     * @param {String} Sentence
     * @return {Tensor} a one-hot array of vectorized letter sequence
     */
    fivebit (sentence) {
        return [].concat.apply( [], sentence.split('').map( letter =>
            Tensor(5, i => (this.clmap[letter]+1) >> i & 1)
        ) );
    }
}

// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
// NeuralNet
// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class NeuralNet {
    /**
     * Initialize NeuralNet.
     *
     * @param {Object} setup vaules for the neuralNet
     * @return {Object} this
     */
    constructor (setup) {
        this.setup        = setup = setup      || {};
        this.learn        = setup.learn        || 0.001;
        this.type         = setup.type         || 'standard';
        this.layers       = setup.layers       || 0;
        this.epochs       = setup.epochs       || 2000;
        this.batchSize    = setup.batchSize    || 10;
        this.losses       = [];
    }

    /**
     * Train the Neural Network.
     *
     * @param {Object} training values and parameters
     */
    train (setup) {
        let dataset    = setup.dataset;
        let batchSize  = setup.batchSize || this.batchSize;
        let inputSize  = dataset[0][0].length;
        let outputSize = dataset[0][1].length;
        let epochs     = setup.epochs || this.epochs;
        let now        = +new Date;

        // Generate layers
        this.layers = this.layers || Layer.generate(
            inputSize, outputSize, this.type
        );

        // Train epochs in random batches
        Array(epochs).fill().forEach( (_, epoch) => {
            let batch   = Batcher({ dataset : dataset, size : batchSize })
            let inputs  = batch.map(a => a[0]);
            let targets = batch.map(a => a[1]);

            // Forward propagate inputs without modifying tensors
            let predicted = this.forward(inputs);

            // Calculate Loss and Gradient
            let loss  = math.mse.loss(predicted, targets);
            let grads = math.mse.grad(predicted, targets);

            // Save Losses for Charting
            this.losses.push(loss);

            // Show Progress for long training sessions
            if (+new Date - now > 1000) {
                now = +new Date;
                let cost = this.losses.reduce((a,b)=>a+b)/this.losses.length;
                console.log(
                    'training', (epoch + '').padEnd(6)
                ,   'cost', cost
                );
            }

            // Create Gradients for Training
            grads = this.backward(grads);

            // Learn/SGD
            // Update Weights and Bias with Learnings for Gradients
            this.optimize();
        });

        return this;
    };

    /**
     * How big is the AI in Bits and Bytes?
     *
     * @return {Number} size in KB of matrix
     */
    matrixSize () {
        return Math.round(this.save().length/1024) + 'KB';
    };

    /**
     * Ask the AI a question.
     *
     * @param {Tensor} array of inputs for predicting outputs
     * @return {Tensor} array of answer outputs
     */
    predict (features) {
        return this.forward(features);
    };

    /**
     * Load Layers from a JSON String.
     *
     * @param {String} JSON string data
     */
    load (json) {
        this.layers = JSON.parse(json);
    }

    /**
     * Save Layers as a JSON String.
     *
     * @return {String} JSON ouput string
     */
    save () {
        this.layers.forEach(layer =>
            ['inputs', 'gradients'].map(p => delete layer[p])
        );
        return JSON.stringify(this.layers);
    };

    /**
     * Forward Propgegation.
     * Inputs are a batch size, outputs are predictions from batch input.
     * 
     * @param {Tensor} array of inputs for predicting outputs
     * @return {Tensor} array of answer outputs
     */
    forward (inputs) {
        let output = null; 
        //console.log('inputs',inputs);
        //inputs = inputs.map(a => a.concat([1]));
        //console.log('inputs-bias',inputs);

        this.layers.forEach( layer => {
            layer.inputs = output || inputs;
            output = math.mmul(
                layer.inputs,
                layer.parameters.weights
            ).map(a => math.add(a, layer.parameters.bias)
            ).map(a => a.map(math[layer.activation]));
        });

        return output;
    }

    /**
     * Backward Propgegation.
     * Generate gradient weights and bias from delta of targets and output.
     * 
     * @param {Tensor} gradient array of inputs for predicting outputs
     * @return {Tensor} gradient array of answer outputs
     */
    backward (gradient) {
        this.layers.slice().reverse().forEach( layer => {
            layer.gradients.bias    = math.sum(gradient[0]);
            layer.gradients.weights = math.mmul(math.transpose(layer.inputs), gradient);
            gradient = math.mmul(gradient, math.transpose(layer.parameters.weights));
            gradient = gradient.map( (a, i) => a.map( (b, k) => b * math[layer.derivative](layer.inputs[i][k]) ) );
        });

        return gradient;
    }

    /**
     * Optimze.
     * Apply gradient learnings gradually based on learning rate.
     */
    optimize() {
        this.layers.forEach( layer => {
            // Update weights
            layer.parameters.weights = layer.parameters.weights
                  .map((pw, x) =>
                pw.map((p,  y) => 
                    p - layer.gradients.weights[x][y] * this.learn
            ));

            // Update bias
            layer.parameters.bias = layer.parameters.bias.map(
                b => b - layer.gradients.bias * this.learn
            );
        });
    }
}

// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
// Math
// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class math {}

math.reLU      = (val)  => val * (val > 0);
math.reLUD     = (val)  => 1. * (math.reLU(val) > 0);
math.reLUP     = (val)  => 1. * (val > 0);
math.sigmoid   = (val)  => 1/(1+Math.pow(Math.E, -val));
math.sigmoidD  = (val)  => math.sigmoid(val) * (1 - math.sigmoid(val));
math.sigmoidP  = (val)  => val * (1 - val)
math.tanh      = (val)  => Math.tanh(val)
math.tanhD     = (val)  => 1 - math.tanh(val) ** 2;
math.tanhP     = (val)  => 1 - val ** 2;
math.linear    = (val)  => val;
math.linearP   = (val)  => val;
math.transpose = (val)  => val[0].map((x, i) => val.map((y, k) => y[i]));
math.dot       = (a, b) => a.map((x, i) => x * b[i]).reduce((m, n) => m + n);
math.mmul      = (a, b) => a.map(x => math.transpose(b).map(y => math.dot(x, y)));
math.sum       = (a   ) => a.reduce( (x, y) => x + y );
math.add       = (a, b) => a.map((x, i) => b.map((y, n) => x + y).reduce((x, y) => x + y ));

// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
// Math Mean Squared Error for Loss and Gradient Error
// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
math.mse = {};
math.mse.grad = (predicted, targets) =>
    predicted.map( (p, n) => p.map( (p, k) => 2 * (p - targets[n][k]) ));
math.mse.loss = (predicted, targets) => 
    math.sum(predicted.map( (p, n) =>
        math.sum(p.map( (p, k) => (p - targets[n][k]) ** 2 ))
    ));

// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
// Tensor
// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
const Tensor = (size, fn) => {
    let rn = () => 2*Math.random() - 2*Math.random();
    return Array.from(Array(size || 1), (j, i) => fn ? fn(i) : rn());
};

// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
// Training Batches
// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
const Batcher = (setup) => {
    let data  = setup.dataset || console.error("Missing Training Data") || [];
    let size  = setup.size    || 1;
    let len   = data.length;
    return Tensor(size, i => data[Math.round(Math.random()*(len-1))]);
};

// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
// Layer
// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
const Layer = (setup) => {
    let count      = Layer.count = (Layer.count || 0) + 1;
    let activation = setup.activation || "tanh";
    let name       = setup.name       || "Layer " + count;
    let input      = setup.input      || 2;
    let output     = setup.output     || 1;
    let weights    = Tensor(input,  () => Tensor(output));
    let bias       = Tensor(output);

    return {
        name       : name
    ,   number     : count
    ,   activation : activation     // forward  propagation
    ,   derivative : activation+"P" // backward propagation
    ,   parameters : { bias: bias, weights: weights }
    ,   gradients  : { bias:   [], weights: [] }
    };
};

Layer.generate = (input, output, type='standard', density=3) => {
    let dense   = density * input;
    let half    = Math.ceil(input/2);
    let quarter = Math.ceil(input/4);
    let tenth   = Math.ceil(input/10);

    return ({
        standard : () => [
            Layer({ input: input, output: dense,  activation: 'tanh'   })
        ,   Layer({ input: dense, output: output, activation: 'linear' })],
        tanh : () => [
            Layer({ input: input, output: dense,  activation: 'tanh'   })
        ,   Layer({ input: dense, output: output, activation: 'linear' })],
        sigmoid : () => [
            Layer({ input: input, output: dense,  activation: 'sigmoid' })
        ,   Layer({ input: dense, output: output, activation: 'linear'  })],
        reLU : () => [
            Layer({ input: input, output: dense,  activation: 'reLU'   })
        ,   Layer({ input: dense, output: output, activation: 'linear' })],
        deep : () => [
            Layer({ input: input, output: dense,  activation: 'tanh'   })
        ,   Layer({ input: dense, output: dense,  activation: 'tanh'   })
        ,   Layer({ input: dense, output: dense,  activation: 'tanh'   })
        ,   Layer({ input: dense, output: output, activation: 'linear' })],
        text : () => [
            Layer({ input: input,   output: dense,   activation: 'tanh'   })
        ,   Layer({ input: dense,   output: half,    activation: 'tanh'   })
        ,   Layer({ input: half,    output: quarter, activation: 'tanh'   })
        ,   Layer({ input: quarter, output: tenth,   activation: 'tanh'   })
        ,   Layer({ input: tenth,   output: output,  activation: 'linear' })],
    })[type]();
};

// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
// Exports
// =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
module.exports = { NeuralNet, Vectorizer, Importer, Visualizer, Tensor };
