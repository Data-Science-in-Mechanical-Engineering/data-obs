function [] = generateTestData()
    params.sig = -1;
    params.shuff = 1000;
    params.bootForce = 1;

    generateTestCase01(params);
    generateTestCase02(params);
    generateTestCase03(params);
    generateTestCase04(params);
    generateTestCase05(params);
end

function [] = generateTestCase01(params)
    rng(1)

    m = 2;
    n = 2;
    dimensions = 1;

    muX = zeros(dimensions, 1);
    sigmaX = eye(dimensions);

    muY = ones(dimensions, 1);
    sigmaY = eye(dimensions);

    X = mvnrnd(muX, sigmaX, m);
    Y = mvnrnd(muY, sigmaY, n);

    mmdTestBoot(X, Y, 0.05, params, "01");
end

function [] = generateTestCase02(params)
    rng(1)

    m = 200;
    dimensions = 2;

    muX = zeros(dimensions, 1);
    sigmaX = eye(dimensions);

    X = mvnrnd(muX, sigmaX, m);

    mmdTestBoot(X, X, 0.05, params, "02");
end

function [] = generateTestCase03(params)
    rng(1)

    m = 20;
    n = 20;
    dimensions = 1200;

    muX = zeros(dimensions, 1);
    sigmaX = eye(dimensions);

    muY = ones(dimensions, 1);
    sigmaY = eye(dimensions);

    X = mvnrnd(muX, sigmaX, m);
    Y = mvnrnd(muY, sigmaY, n);

    mmdTestBoot(X, Y, 0.05, params, "03");
end

function [] = generateTestCase04(params)
    rng(1)

    m = 200;
    n = 200;
    dimensions = 10;

    muX = zeros(dimensions, 1);
    sigmaX = eye(dimensions);

    muY = zeros(dimensions, 1);
    sigmaY = 2*eye(dimensions);

    X = mvnrnd(muX, sigmaX, m);
    Y = mvnrnd(muY, sigmaY, n);

    mmdTestBoot(X, Y, 0.05, params, "04");
end

function [] = generateTestCase05(params)
    rng(1)

    m = 200;
    n = 200;
    dimensions = 10;

    muX = zeros(dimensions, 1);
    sigmaX = eye(dimensions);

    muY = ones(dimensions, 1);
    sigmaY = 2*eye(dimensions);

    X = mvnrnd(muX, sigmaX, m);
    Y = mvnrnd(muY, sigmaY, n);

    mmdTestBoot(X, Y, 0.05, params, "05");
end
