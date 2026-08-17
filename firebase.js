import { initializeApp } from "https://www.gstatic.com/firebasejs/10.7.0/firebase-app.js";
import { getFirestore, collection, addDoc, getDocs, doc, deleteDoc, updateDoc } from "https://www.gstatic.com/firebasejs/10.7.0/firebase-firestore.js";
const firebaseConfig = {
  apiKey: "AIzaSyBSKybHC6JO8TrxpEHGhRg64htixzHoExY",
  authDomain: "health-care--hck.firebaseapp.com",
  projectId: "health-care--hck",
  storageBucket: "health-care--hck.firebasestorage.app",
  messagingSenderId: "533950942274",
  appId: "1:533950942274:web:301163a44d963d19ebf1db"
};

const app = initializeApp(firebaseConfig);
const db = getFirestore(app);

window.addReminder = async function() {
  const name = document.getElementById("nameInput").value;
  const time = document.getElementById("timeInput").value;

  try {
    const docRef = await addDoc(collection(db, "reminders"), {
      name: name,
      time: time,
      createdAt: new Date()
    });
    console.log("Added with ID:", docRef.id);
  } catch (e) {
    console.error("Error:", e);
  }
}

window.getReminders = async function() {
  const querySnapshot = await getDocs(collection(db, "reminders"));
  const listElement = document.getElementById("reminderList");
  listElement.innerHTML = "";

  querySnapshot.forEach((doc) => {
    const data = doc.data();
    const li = document.createElement("li");
    li.textContent = `${data.name} - ${data.time} `;

    const editBtn = document.createElement("button");
    editBtn.textContent = "Edit";
   editBtn.onclick = function() {
  const newName = prompt("Enter new name:", data.name);
  const newTime = prompt("Enter new time:", data.time);
  if (newName && newTime) {
    updateReminder(doc.id, newName, newTime);
  }
};

    const deleteBtn = document.createElement("button");
    deleteBtn.textContent = "Delete";
    deleteBtn.onclick = function() {
      deleteReminder(doc.id);
    };

    li.appendChild(editBtn);
    li.appendChild(deleteBtn);
    listElement.appendChild(li);
  });
}
window.deleteReminder = async function(id) {
  try {
    await deleteDoc(doc(db, "reminders", id));
    console.log("Deleted:", id);
    getReminders(); // refresh the list after deleting
  } catch (e) {
    console.error("Error deleting:", e);
  }
}

window.updateReminder = async function(id, newName, newTime) {
  try {
    const reminderRef = doc(db, "reminders", id);
    await updateDoc(reminderRef, { name: newName, time: newTime });
    console.log("Updated:", id);
    getReminders(); // refresh the list
  } catch (e) {
    console.error("Error updating:", e);
  }
}
