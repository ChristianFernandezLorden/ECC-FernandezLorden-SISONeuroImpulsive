# LIF

function lif!(du, u, p, t)
    du[1] = p[2](t) - p[1]*u[1] 
end

function lif_condition(u, t, integrator) 
    return u[1] - integrator.p[3]
end

function lif_up_affect!(integrator)
    integrator.u[1] = integrator.p[4]
    if integrator.t > integrator.p[6]
        append!(integrator.p[5], integrator.t)
    end
end

# IF Pair + linear system

function linear_if!(du, u, p, t)
    du[1] = p[1]*u[1]
    du[2] = max(0, p[8]+p[3]*u[1]) #- p[1]*u[1] 
    du[3] = max(0, p[8]-p[3]*u[1]) #- p[1]*u[1] 
end

function linear_if_condition_1(u, t, integrator) 
    return u[2] - integrator.p[4]
end


function linear_if_condition_2(u, t, integrator) 
    return u[3] - integrator.p[4]
end

function linear_if_up_affect_1!(integrator)
    integrator.u[2] = 0
    integrator.u[1] = integrator.u[1] - integrator.p[2]*integrator.p[5]*abs(integrator.u[1])
    append!(integrator.p[6], integrator.t)
end

function linear_if_up_affect_2!(integrator)
    integrator.u[3] = 0
    integrator.u[1] = integrator.u[1] + integrator.p[2]*integrator.p[5]*abs(integrator.u[1])
    append!(integrator.p[7], integrator.t)
end


# IF Pair Track + linear system

function linear_track_if!(du, u, p, t)
    du[1] = p[1]*u[1] 
    du[2] = max(0, p[8] + p[3]*u[1] - p[9](t)) #- p[1]*u[1] 
    du[3] = max(0, p[8] - p[3]*u[1] + p[9](t)) #- p[1]*u[1] 
end

function linear_track_if_condition_1(u, t, integrator) 
    return u[2] - integrator.p[4]
end


function linear_track_if_condition_2(u, t, integrator) 
    return u[3] - integrator.p[4]
end

function linear_track_if_up_affect_1!(integrator)
    integrator.u[2] = 0
    integrator.u[1] = integrator.u[1] - integrator.p[2]*integrator.p[5]*abs(integrator.u[1]- p[9](integrator.t))
    append!(integrator.p[6], integrator.t)
end

function linear_track_if_up_affect_2!(integrator)
    integrator.u[3] = 0
    integrator.u[1] = integrator.u[1] + integrator.p[2]*integrator.p[5]*abs(integrator.u[1]- p[9](integrator.t))
    append!(integrator.p[7], integrator.t)
end



# IF Pair + CB=0 linear system

function linear0_if!(du, u, p, t)
    n = size(p[1],1)
    du[3:2+n] = p[1]*u[3:2+n]
    du[1] = max(0, p[8] + norm(p[3] * u[3:2+n])) - p[9]*u[1]
    du[2] = max(0, p[8] - norm(p[3] * u[3:2+n])) - p[9]*u[2]
end

function linear0_if_condition_1(u, t, integrator) 
    return u[1] - integrator.p[4]
end


function linear0_if_condition_2(u, t, integrator) 
    return u[2] - integrator.p[4]
end

function linear0_if_up_affect_1!(integrator)
    n = size(integrator.p[1],1)
    integrator.u[1] = 0
    integrator.u[3:2+n] = (I - integrator.p[5] * integrator.p[3] * integrator.p[2])*integrator.u[3:2+n]
    append!(integrator.p[6], integrator.t)
end

function linear0_if_up_affect_2!(integrator)
    n = size(integrator.p[1],1)
    integrator.u[2] = 0
    integrator.u[3:2+n] = (I - integrator.p[5] * integrator.p[3] * integrator.p[2])*integrator.u[3:2+n]
    append!(integrator.p[7], integrator.t)
end



# Integrator

function if_integrator!(du, u, p, t)
    du[1] = max(0, p[5] + p[1](t)) 
    du[2] = max(0, p[5] - p[1](t))
    du[3] = max(0, p[6])
    du[4] = max(0, -p[6])
    du[5] = p[1](t)
end

function if_integrator_condition_1(u, t, integrator) 
    return u[1] - integrator.p[2]
end


function if_integrator_condition_2(u, t, integrator) 
    return u[2] - integrator.p[2]
end

function if_integrator_condition_3(u, t, integrator)
    return u[3] - integrator.p[3]
end


function if_integrator_condition_4(u, t, integrator)
    return u[4] - integrator.p[3]
end

function if_integrator_up_affect_1!(integrator)
    integrator.u[1] = 0
    integrator.p[6] = integrator.p[6] + integrator.p[4]
    append!(integrator.p[7], integrator.t)
end

function if_integrator_up_affect_2!(integrator)
    integrator.u[2] = 0
    integrator.p[6] = integrator.p[6] - integrator.p[4]
    append!(integrator.p[8], integrator.t)
end

function if_integrator_up_affect_3!(integrator)
    integrator.u[3] = 0
    append!(integrator.p[9], integrator.t)
end

function if_integrator_up_affect_4!(integrator)
    integrator.u[4] = 0
    append!(integrator.p[10], integrator.t)
end