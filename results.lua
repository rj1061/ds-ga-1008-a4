stringx = require('pl.stringx')
require 'io'
require 'cunn'
require 'nn'
require 'nngraph'
require('./base')
ptb = require('data')

model = torch.load('model.t7')

gpu = true
if gpu then
    require 'cunn'
    print("Running on GPU")

else
    require 'nn'
    print("Running on CPU")
end

local params = {
                batch_size=20, -- minibatch
                seq_length=20, -- unroll length
                layers=2,
                decay=2,
                rnn_size=200, -- hidden unit size
                dropout=0.2,
                init_weight=0.1, -- random weight initialization limits
                lr=0.9, --learning rate
                vocab_size=10000, -- limit on the vocabulary size
                max_epoch=4,  -- when to start decaying learning rate
                max_max_epoch=8, -- final epoch
                max_grad_norm=5 -- clip when gradients exceed this norm value
               }


function transfer_data(x)
    if gpu then
        return x:cuda()
    else
        return x
    end
end

function reset_state(state)
    state.pos = 1
    if model ~= nil and model.start_s ~= nil then
        for d = 1, 2 * params.layers do
            model.start_s[d]:zero()
        end
    end
end

function reset_ds()
    for d = 1, #model.ds do
        model.ds[d]:zero()
    end
end

function fp(state)
    -- g_replace_table(from, to).
    g_replace_table(model.s[0], model.start_s)

    -- reset state when we are done with one full epoch
    if state.pos + params.seq_length > state.data:size(1) then
        reset_state(state)
    end

    -- forward prop
    for i = 1, params.seq_length do
        local x = state.data[state.pos]
        local y = state.data[state.pos + 1]
        local s = model.s[i - 1]
        model.err[i], model.s[i], p = unpack(model.rnns[i]:forward({x, y, s}))
        state.pos = state.pos + 1
    end

    -- next-forward-prop start state is current-forward-prop's last state
    g_replace_table(model.start_s, model.s[params.seq_length])

    -- cross entropy error
    return model.err:mean(), p
end


function run_valid()
    -- again start with a clean slate
    reset_state(state_valid)

    -- no dropout in testing/validating
    g_disable_dropout(model.rnns)

    -- collect perplexity over the whole validation set
    local len = (state_valid.data:size(1) - 1) / (params.seq_length)
    local perp = 0
    for i = 1, len do
        perp = perp + fp(state_valid)
    end
    print("Validation set perplexity : " .. g_f3(torch.exp(perp / len)))
    g_enable_dropout(model.rnns)
end

function run_test()
    reset_state(state_test)
    g_disable_dropout(model.rnns)
    local perp = 0
    local len = state_test.data:size(1)

    -- no batching here
    g_replace_table(model.s[0], model.start_s)
    for i = 1, (len - 1) do
        local x = state_test.data[i]
        local y = state_test.data[i + 1]
        perp_tmp, model.s[1] = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
        perp = perp + perp_tmp[1]
        g_replace_table(model.s[0], model.s[1])
    end
    print("Test set perplexity : " .. g_f3(torch.exp(perp / (len - 1))))
    g_enable_dropout(model.rnns)
end

state_valid =  {data=transfer_data(ptb.validdataset(params.batch_size))}
state_test =  {data=transfer_data(ptb.testdataset(params.batch_size))}

print("Network parameters:")
print(params)

local states = {state_train, state_valid, state_test}
for _, state in pairs(states) do
    reset_state(state)
end

print("Testing results")
run_test()
print("Validation results")
run_valid()
print("--------------------")
