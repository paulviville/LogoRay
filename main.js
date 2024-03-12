'use strict';

import Stats from './stats.module.js';

const stats = new Stats()
document.body.appendChild( stats.dom );

const canvas = document.getElementById("gl-canvas");
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
console.log(canvas);
const gl = canvas.getContext('webgl2');
console.log(gl);


gl.viewport(0, 0, canvas.width, canvas.height);

const vertexShader = `#version 300 es
precision lowp float;

in vec2 vPos;
uniform float ratio;

out vec2 uv;
void main()
{
if(ratio > 1.)
      uv = vec2(vPos.x* ratio, vPos.y );
else
  uv = vec2(vPos.x, vPos.y / ratio);

    gl_Position = vec4(vPos, 0.0, 1.0);
}
`;

const fragmentShader = `#version 300 es
precision lowp float;
uniform float ratio;
in vec2 uv;
out vec4 fragColor;
uniform float time;

float sdSphere( vec3 p, vec4 s){
    return length(s.xyz - p) - s.w;
}

float sdPlane( vec3 p, vec3 n, float h ){
    return dot(p,n) + h;
}

float sdTorus( vec3 p, vec2 t )
{
    vec2 q = vec2(length(p.yx)-t.x,p.z);
    return length(q)-t.y;
}

float sdCone( vec3 p, vec2 c, float h )
{
  // c is the sin/cos of the angle, h is height
  // Alternatively pass q instead of (c,h),
  // which is the point at the base in 2D
  vec2 q = h*vec2(c.x/c.y,-1.0);
    
  vec2 w = vec2( length(p.xz), p.y );
  vec2 a = w - q*clamp( dot(w,q)/dot(q,q), 0.0, 1.0 );
  vec2 b = w - q*vec2( clamp( w.x/q.x, 0.0, 1.0 ), 1.0 );
  float k = sign( q.y );
  float d = min(dot( a, a ),dot(b, b));
  float s = max( k*(w.x*q.y-w.y*q.x),k*(w.y-q.y)  );
  return sqrt(d)*sign(s);
}

float opSmoothUnion( float d1, float d2, float k ) {
    float h = clamp( 0.5 + 0.5*(d2-d1)/k, 0.0, 1.0 );
    return mix( d2, d1, h ) - k*h*(1.0-h); }

float opUnion( float d1, float d2 ) { return min(d1,d2); }

float opIntersection( float d1, float d2 )
{
    return max(d1,d2);
}

float opSmoothIntersection( float d1, float d2, float k )
{
    float h = clamp( 0.5 - 0.5*(d2-d1)/k, 0.0, 1.0 );
    return mix( d2, d1, h ) + k*h*(1.0-h);
}

const float pi = 3.14159265358979323846264338327950288419716939937510582;
vec2 rotate(vec2 p, float angle) {
    float s = sin(angle);
    float c = cos(angle);
    mat2 m = mat2(c, -s, s, c);
    return m * p;
}

vec3 translate(vec3 p, vec3 t) {
    return p - t;
}

const float alpha = pi / 16.;
const float beta = pi / 2. - alpha;
const float gamma = 2. * alpha;

const float radius = 2.;
const float a = sin(alpha) * radius;
const float b = cos(alpha) * radius;
const float c = b * b / a;
const float coneHeight = c;
const vec2 coneAngles = vec2(sin(alpha), cos(alpha));

const vec4 s0 = vec4(0., 0., radius * 0.6, radius);
const vec4 s1 = vec4(0., 0., -radius * 0.6, radius);


float spike(vec3 p) {
    float dist = 1./0.;

    vec3 up = vec3(0., 4. * radius, 0.);
    vec3 p_ = translate(p, up) ;
    float distCone0 = sdCone(p_ + vec3(0., -coneHeight - a, -radius * 0.6), coneAngles, coneHeight);
    float distCone1 = sdCone(p_ + vec3(0., -coneHeight - a, radius * 0.6), coneAngles, coneHeight);

    float distSphere0 = sdSphere(p_, s0);
    float distSphere1 = sdSphere(p_, s1);

    float distSpheres = opIntersection(distSphere0, distSphere1);
    float distCones = opIntersection(distCone0, distCone1);

    dist = min(dist, distSpheres);
    dist = min(dist, distCones);

    return dist;
}

float spoke(vec3 p) {
    float dist = 1./0.;

    vec2 p2 = rotate(p.xy, pi);
    vec3 pr = vec3(p2.x, p2.y, p.z);

    float spike0 = spike(p);
    float spike1 = spike(pr);

    dist = min(dist, spike0);
    dist = min(dist, spike1);

    return dist;
}

const int nbSpokes = 6;
float Distance(vec3 p) {
    float dist = 1./0.;

    vec2 p2 = rotate(p.yz, 4./10.);
    p = vec3(p.x, p2.x, p2.y);

    p2 = rotate(p.xy, - time / 4.);
    p = vec3(p2.x, p2.y, p.z);

    // p2 = rotate(p.yz, - time / 8.);
    // p = vec3(p.x, p2.x, p2.y);

    vec3 torusCenter0 = vec3(0., 0., 0.15);
    vec3 torusCenter1 = vec3(0., 0., -0.15);
    vec3 pTorus0 = p;
    pTorus0 = translate(pTorus0, torusCenter0);

    vec3 pTorus1 = p;
    pTorus1 = translate(pTorus1, torusCenter1);


    float distTorus0 = sdTorus(pTorus0, vec2(1.0, 0.3));
    float distTorus1 = sdTorus(pTorus1, vec2(1.0, 0.3));
    float distTorusInter = opIntersection(distTorus0, distTorus1);
    dist = min(dist, distTorusInter);


    float scale = .25;
    float scaleFactor = 1.2;

    for(int i = 0; i < nbSpokes; ++i) {
        vec3 pr = vec3(rotate(p.xy, float(i - 2) * pi / 5.5), p.z);
        float s = scale * pow(scaleFactor, float(i));
        float d = spoke(pr / s) * s;

        dist = min(dist, d);
    }

    return dist;
}

#define EPS 0.001
const vec2 eps = vec2(EPS, 0.0);
vec3 Gradient(vec3 p) {
    float d = Distance(p);
    vec2 eps = vec2(EPS, -EPS);
    vec3 n = eps.xyy * Distance(p+eps.xyy) + eps.yyx * Distance(p+eps.yyx)+ eps.yxy * Distance(p+eps.yxy)+ eps.xxx * Distance(p+eps.xxx); 
    return normalize(n);
}

#define STEPS 200
#define DIST 50.
#define HIT .0001



float RayMarch(vec3 ro, vec3 rd) {
    float d0 = 0.;

    for(int i = 0; i < STEPS; ++i) {
        vec3 p = ro + rd*d0;
        float dist = Distance(p);
        d0 += dist;
        if(d0 > DIST || abs(dist) < HIT)
            break;
    }

    return d0;
}

const vec3 lightPos0 = vec3(5., 5., -5.);
const vec3 lightPos1 = vec3(5., 5., -5.);
float Light(vec3 p) {
    vec3 l0 = normalize(lightPos0 - p);
    vec3 l1 = normalize(lightPos0 - p);
    vec3 n = Gradient(p);
    vec3 r0 = lightPos0 - p;
    vec3 r1 = lightPos1 - p;
    float d0 = RayMarch(p + .01*normalize(r0), l0);
    float g = (dot(n, l0)+dot(n,l1))/2.;
    g = max(g, 0.1);
    if(d0 < length(r0))
        return g*0.5;
    return g;
}

void main(){
    vec3 color = vec3(0.);

    // camera
    // vec3 ro = vec3(-15., 0., 0.);
    // vec3 rd = normalize(vec3(1, uv));

    vec3 ro = vec3(0., 0., -10.);
    vec3 rd = normalize(vec3(uv, 1.));

    float d = RayMarch(ro, rd);
    vec3 p = ro + rd * d;
    float l = Light(p);
    color = normalize(Gradient(p)) - (l*0.1);
    color = vec3(max(l, 0.1));
    
    if(d >= DIST)
        color = vec3(0.08);


    fragColor = vec4(color, 1.0);
}
`;

function rd(factor = 1.5) {
return (factor*(2*Math.random() - 1));
}

function randomizeMaxShader() {
let fstring = fragmentShader;



return fstring;
}


const shader = gl.createProgram();

const vs = gl.createShader(gl.VERTEX_SHADER);
gl.shaderSource(vs, vertexShader);
gl.compileShader(vs);
gl.attachShader(shader, vs);
if(!gl.getShaderParameter(vs, gl.COMPILE_STATUS)){
console.error("error in vs: ", gl.getShaderInfoLog(vs));
}

const fs = gl.createShader(gl.FRAGMENT_SHADER);
gl.shaderSource(fs, randomizeMaxShader());
gl.compileShader(fs);
gl.attachShader(shader, fs);
if(!gl.getShaderParameter(fs, gl.COMPILE_STATUS)){
console.error("error in fs: ", gl.getShaderInfoLog(fs));
}

gl.linkProgram(shader);
if(!gl.getProgramParameter(shader, gl.LINK_STATUS)){
console.error("error linking program: ", gl.getProgramInfoLog(shader));
}

gl.validateProgram(shader);
if(!gl.getProgramParameter(shader, gl.VALIDATE_STATUS)){
console.error("error validating program: ", gl.getProgramInfoLog(shader));
}
gl.useProgram(shader);



let VBO = gl.createBuffer();
let vertices = [-1, 1, -1, -1, 1, 1, 1, -1];
gl.bindBuffer(gl.ARRAY_BUFFER, VBO);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);

let vPosAttrib = gl.getAttribLocation(shader, 'vPos');
gl.vertexAttribPointer(vPosAttrib, 2, gl.FLOAT, gl.FALSE, 2 * Float32Array.BYTES_PER_ELEMENT, 0);
gl.enableVertexAttribArray(vPosAttrib);


window.addEventListener('resize', function() {
if(!canvas) return;
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
gl.viewport(0, 0, canvas.width, canvas.height);
gl.uniform1f(gl.getUniformLocation(shader, 'ratio'), canvas.width / canvas.height);

});

function render (time) {
gl.uniform1f(gl.getUniformLocation(shader, 'ratio'), canvas.width / canvas.height);

gl.uniform1f(gl.getUniformLocation(shader, 'time'), time);
gl.bindVertexArray(null);

gl.clearColor(0.2, 0.2, 0.2, 1.0);
gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}


function update ()
{
	stats.update()

}

function mainloop(time)
{
update();
render(time / 1000);
requestAnimationFrame(mainloop);
}

mainloop();